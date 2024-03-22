#ifndef TREENODE_H
#define TREENODE_H
#include"utils.hpp"
#include"Card.hpp"
#include"ACTION.hpp"
#include"PS.hpp"
struct TreeNode{
    PS ps;
    int id,depth,fa,child_begin,child_end;
    ACTION final_action;
    TreeNode(){
        child_begin=0;
        child_end=-1;
    };
    TreeNode(PS _ps,int _id,int _depth,int _fa,ACTION _a){
        ps=_ps;
        id=_id;
        depth=_depth;
        fa=_fa;
        child_begin=0;
        child_end=-1;
        final_action=_a;
    }
    TreeNode &operator=(const TreeNode&_nd){
        if(&_nd==this)return *this;
        ps=_nd.ps;
        id=_nd.id;
        depth=_nd.depth;
        fa=_nd.fa;
        child_begin=_nd.child_begin;
        child_end=_nd.child_end;
        final_action=_nd.final_action;
        return *this;
    }
};

#endif