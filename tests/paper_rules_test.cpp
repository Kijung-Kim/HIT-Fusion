#include "../paper_rules.h"
#include <cassert>
#include <iostream>
int main() {
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
