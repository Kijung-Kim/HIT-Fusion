#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// Manuscript Section III-G. Indices are stored-record indices, not filenames.
namespace hit {
// Revised voting rule: equal peak per candidate; bandwidth still depends on size.
inline double unitPeakVote(double dx,double dy,double diameter) {
    if(!std::isfinite(diameter) || diameter<=0) return 0;
    const double sigma=40.0/diameter;
    return std::exp(-(dx*dx+dy*dy)/(2*sigma*sigma));
}
inline int selectCandidate(const std::vector<int>& s,const std::vector<double>& m,int lo,int hi,bool fused=false) {
    if(lo<0 || hi<lo || hi>=int(s.size()) || s.size()!=m.size()) return -1;
    int smin=std::numeric_limits<int>::max(),smax=0;
    double mmin=std::numeric_limits<double>::infinity(),mmax=0;
    for(int i=lo;i<=hi;++i) {
        if(s[i]<0 || !std::isfinite(m[i]) || m[i]<0) return -1;
        if(s[i]>0) {smin=std::min(smin,s[i]);smax=std::max(smax,s[i]);}
        if(m[i]>0) {mmin=std::min(mmin,m[i]);mmax=std::max(mmax,m[i]);}
    }
    if(smax==0 || mmax==0) return -1;
    double sgap=std::numeric_limits<double>::infinity(),mgap=sgap,scut=0,mcut=0;
    // Ascending scan with <= selects the latest tied minimizer.
    for(int i=lo;i<=hi;++i) if(s[i]>0) {
        double gs=std::abs((double(s[i])-smin)-(smax-double(s[i])));
        if(gs<=sgap) {sgap=gs;scut=0.5*s[i];}
        double gm=std::abs((m[i]-mmin)-(mmax-m[i]));
        if(gm<=mgap) {mgap=gm;mcut=m[i];}
    }
    int best=-1;
    for(int i=lo;i<=hi;++i) if(s[i]>=scut && m[i]>=mcut)
        if(best<0 || (fused?s[i]*m[i]:m[i])>(fused?s[best]*m[best]:m[best])) best=i;
    return best;
}
inline int stableFrame(const std::vector<int>& s,int candidate,int direction,int lo,int hi) {
    int run=0;
    for(int offset=0;offset<20;++offset) {
        int i=candidate+direction*offset;
        if(i<lo || i>hi || i<0 || i>=int(s.size())) break;
        run=(s[i]==0)?run+1:0;
        if(run==3) return i;
    }
    return -1;
}
inline int refine(int candidate,const std::vector<int>& heights) {
    int best=-1;
    for(int i=std::max(0,candidate-4);i<=std::min(int(heights.size())-1,candidate+4);++i) {
        if(heights[i]<0) continue;
        if(best<0 || heights[i]>heights[best] ||
           (heights[i]==heights[best] && std::abs(i-candidate)<std::abs(best-candidate))) best=i;
    }
    return best<0?candidate:best;
}
inline int stableNeighborFrame(const std::vector<int>& s,int candidate,int direction,int lo,int hi,int maxChange=0) {
    int run=0;
    for(int offset=1;offset<=20;++offset) {
        int i=candidate+direction*offset;
        if(i<lo || i>hi || i<0 || i>=int(s.size())) break;
        run=(s[i]>=0 && s[i]<=maxChange)?run+1:0;
        if(run==3) return i;
    }
    return -1;
}
inline int eventOnset(const std::vector<int>& s,int verified,int lo) {
    if(verified<lo || verified<0 || verified>=int(s.size()) || s[verified]<=0) return verified;
    int onset=verified;
    for(int i=verified-1;i>=std::max(lo,verified-4);--i) {
        if(s[i]<0 || 2*s[i]<s[verified]) break;
        onset=i;
    }
    return onset;
}
}
