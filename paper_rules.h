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
inline int selectCandidate(const std::vector<int>& s,const std::vector<double>& m,int lo,int hi) {
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
        if(best<0 || m[i]>m[best]) best=i; // earliest equal maximum
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
}
