// Reuse established enhancement and optical-flow routines without the GUI driver.
#define main legacyMain
#include "golfBall.cpp"
#undef main
#include "paper_rules.h"
#include <chrono>
#include <memory>

int main(int argc,char** argv) {
    if(argc!=5 && argc!=6) {
        std::cerr<<"Usage: hit_paper.exe FRAMES_DIR SEGMENTATION.yml RESULT.yml SIGNALS.csv [v3|v4|v4_ball|v4_temporal|v4_occlusion]\n";
        return 2;
    }
    const auto start=std::chrono::steady_clock::now();
    std::string status="error",message;
    int prediction=-1,candidate=-1,verified=-1; cv::Point ball(-1,-1); int frames=0;
    try {
        cv::setNumThreads(1); std::srand(0); cv::setRNGSeed(0);
        std::string folder=argv[1]; if(folder.back()!='/' && folder.back()!='\\') folder+='/';
        auto files=loadPath(folder,"jpg");
        // Numeric filename order is the authoritative time axis.
        auto number=[](const std::string& path) {
            size_t begin=path.find_last_of("/\\");
            return std::stoi(path.substr(begin==std::string::npos?0:begin+1));
        };
        std::sort(files.begin(),files.end(),[&](const std::string& a,const std::string& b){return number(a)<number(b);});
        frames=int(files.size());
        if(frames<12 || frames>=m_result_Max) throw std::runtime_error("unsupported_sequence_length");
        for(int i=1;i<frames;++i) if(number(files[i])<=number(files[i-1])) throw std::runtime_error("duplicate_frame_id");
        auto pose=std::unique_ptr<kGolfPose>(new kGolfPose("yolo.cfg","yolo.weights"));
        auto& p=*pose; p.initResult(); p.m_paperMaskPath=argv[2];
        std::string profile=argc==6?argv[5]:"v4";
        if(profile!="v3" && profile!="v4" && profile!="v4_ball" && profile!="v4_temporal" && profile!="v4_occlusion") throw std::runtime_error("unknown_profile");
        p.m_v4Ball=profile=="v4" || profile=="v4_ball";
        p.m_v4Temporal=profile=="v4" || profile=="v4_temporal";
        p.m_v4Occlusion=profile=="v4" || profile=="v4_occlusion";
        std::cout<<"AUDIT profile "<<profile<<std::endl;
        p.totalImgNumMember=frames-1; p.m_resultsNum=frames-1;
        Mat first=imread(files[std::min(10,frames-1)],1);
        if(first.empty() || first.size()!=Size(1920,1080)) throw std::runtime_error("expected_FHD_frames");
        if(!p.kGolfPreProcessing(first)) {status=p.m_failureReason;}
        else {
            // Keep raw grayscale frames for BGS/flow. Ball channels use white balance.
            std::vector<Mat> gray(frames);
            for(int i=0;i<frames;++i) {
                Mat color=imread(files[i],1);
                if(color.empty() || color.size()!=first.size()) throw std::runtime_error("invalid_frame");
                cvtColor(color,gray[i],CV_BGR2GRAY);
                if(p.m_v4Ball) p.m_results[i].curImg=color(p.m_ballRegion).clone();
                if(i>=1 && i<=120) {
                    balance_white(color);
                    Mat enhanced=color(p.m_ballRegion).clone();
                    // 개선버전 4
                    if(p.m_v4Ball) {
                        Mat raw[3];split(enhanced,raw);
                        for(int c=0;c<3;++c) {
                            Mat full(first.size(),CV_8UC1,Scalar(0));raw[c].copyTo(full(p.m_ballRegion));
                            if(c==0) p.m_results[i].curBlueImg=full;
                            if(c==1) p.m_results[i].curGreenImg=full;
                            if(c==2) p.m_results[i].curRedImg=full;
                        }
                        p.blobDetectionIdx(enhanced,i);
                    }
                    p.imgEnhance(enhanced,enhanced);
                    // Feed the enhanced ROI into extraction. The legacy driver computed
                    // Retinex but discarded it and extracted from the original channels.
                    Mat bgr[3];split(enhanced,bgr);
                    for(int c=0;c<3;++c) {
                        Mat full(first.size(),CV_8UC1,Scalar(0));bgr[c].copyTo(full(p.m_ballRegion));
                        if(c==0) p.m_results[i].curBlueImg=full;
                        if(c==1) p.m_results[i].curGreenImg=full;
                        if(c==2) p.m_results[i].curRedImg=full;
                    }
                    p.blobDetectionIdx(enhanced,i);
                    std::cout<<"AUDIT candidates "<<i<<" "<<p.m_results[i].paperCandidates.size()<<std::endl;
                    p.m_results[i].curBlueImg.release();p.m_results[i].curRedImg.release();p.m_results[i].curGreenImg.release();
                }
            }
            for(int i=1;i<frames;++i) {
                auto& r=p.m_results[i];r.preGrayImg=gray[i-1];r.curGrayImg=gray[i];
                if(i+1<frames) r.fuGrayImg=gray[i+1];
                p.kFlowEstimationIdx(i);
            }
            if(!p.findBallPosition()) status=p.m_failureReason;
            else {
                ball=p.m_ballPosition;candidate=p.estimationImpactSide();verified=p.m_impactIdx;
                if(candidate<0) status=p.m_failureReason;
                else {
                    // 개선버전 4
                    int refined=candidate;
                    std::cout<<"AUDIT verified_candidate_output "<<candidate<<std::endl;
                    prediction=number(files[refined]);status="impact";
                }
            }
            std::ofstream signals(argv[4]);
            if(!signals) throw std::runtime_error("cannot_write_signals");
            signals<<"record_index,filename_frame_id,local_change,motion_energy,endpoint_y\n";
            for(int i=1;i<frames;++i) signals<<i<<','<<number(files[i])<<','<<p.m_results[i].localImpactBGSCount<<','
                <<p.m_results[i].flowEnergySum<<','<<(p.m_results[i].ckSkipEndPoint?-1:p.m_results[i].endP.y)<<'\n';
        }
    } catch(const std::exception& e) {status="error";message=e.what();}
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    FileStorage output(argv[3],FileStorage::WRITE);
    if(!output.isOpened()) return 3;
    output<<"status"<<status<<"error"<<message<<"prediction_frame_id"<<prediction
          <<"candidate_record_index"<<candidate<<"verified_anchor_record_index"<<verified<<"ball_x"<<ball.x<<"ball_y"<<ball.y
          <<"frames"<<frames<<"pipeline_seconds"<<seconds;
    std::cout<<"RESULT "<<status<<" frame="<<prediction<<" seconds="<<seconds<<std::endl;
    return status=="error"?1:0;
}
