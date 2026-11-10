#include "../paper_rules.h"
#include <cassert>
#include <iostream>
int main() {
    assert(hit::eventOnset({0,108,148,182,0},3,0)==1);
    assert(hit::eventOnset({0,17,109},2,0)==2);
    assert(hit::eventOnset({100,0,100},2,0)==2);
    assert(hit::eventOnset({50,50,50,50,50,100},5,0)==1);
    assert(hit::eventOnset({100,100,100},2,2)==2);
    assert(hit::eventOnset({0,0},1,0)==1);
    assert(hit::selectCandidate({2,8,5},{1,7,9},0,2)==2);
    assert(hit::selectCandidate({2,8,5},{1,7,9},0,2,true)==1);
    assert(hit::stableNeighborFrame({2,0,13,13},0,1,0,3,13)==3);
    assert(hit::stableNeighborFrame({2,0,14,13},0,1,0,3,13)==-1);
    assert(hit::stableNeighborFrame({2,-1,0,0},0,1,0,3,13)==-1);
    std::vector<int> boundary(22,1);
    boundary[18]=boundary[19]=boundary[20]=0;
    assert(hit::stableFrame(boundary,0,1,0,21)==-1);
    assert(hit::stableNeighborFrame(boundary,0,1,0,21)==20);
    assert(hit::stableNeighborFrame(boundary,0,1,0,19)==-1);
    boundary[18]=1;boundary[21]=0;
    assert(hit::stableNeighborFrame(boundary,0,1,0,21)==-1);
    assert(hit::stableNeighborFrame({0,0,0,2},3,-1,0,3)==0);
    assert(hit::stableNeighborFrame({0,0,0},0,1,0,2)==-1);
    for(double diameter : {7.0,10.0,20.0,40.0}) {
        assert(hit::unitPeakVote(0,0,diameter)==1.0);
        assert(std::abs(hit::unitPeakVote(40.0/diameter,0,diameter)-std::exp(-0.5))<1e-12);
    }
    assert(hit::unitPeakVote(0,0,0)==0);
    assert(std::abs(hit::unitPeakVote(3,4,10)-hit::unitPeakVote(-3,-4,10))<1e-12);
    // Latest midpoint representative, earliest equal eligible maximum.
    assert(hit::selectCandidate({2,4,6,4},{1,8,9,8},0,3)==2);
    assert(hit::selectCandidate({4,4,4},{8,8,8},0,2)==0);
    assert(hit::selectCandidate({0,0},{1,2},0,1)==-1);
    assert(hit::selectCandidate({1,2},{0,0},0,1)==-1);
    assert(hit::selectCandidate({1,-1},{1,2},0,1)==-1);
    // A fractional Scut must not be truncated: 0.5 * 3 = 1.5.
    assert(hit::selectCandidate({1,3,3},{10,5,5},0,2)==1);
    assert(hit::stableFrame({0,0,0,2,0,0,0},3,-1,0,6)==0);
    assert(hit::stableFrame({0,0,0,2,0,0,0},3,1,0,6)==6);
    assert(hit::stableFrame({0,0,0,2,0,0,0},3,-1,1,6)==-1);
    assert(hit::stableFrame({2,-1,0,0},0,1,0,3)==-1);
    // Include the +4 endpoint, skip missing values, and resolve ties by distance.
    assert(hit::refine(0,{-1,-1,-1,-1,10})==4);
    assert(hit::refine(3,{-1,10,-1,-1,-1,10})==1);
    assert(hit::refine(3,{-1,10,-1,10})==3);
    assert(hit::refine(1,{-1,-1,-1})==1);
    std::cout<<"paper rule tests passed\n";
}
