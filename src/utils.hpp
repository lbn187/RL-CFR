#ifndef UTILS_H
#define UTILS_H
#include<bits/stdc++.h>
#include<random>
#define floord(n,d) floor(((double)(n))/((double)(d)))
using namespace std;
const int MAX_PUBLIC_CARDS=5;
const int CARD_NUMBER=52;
const int HANDS_NUMBER=1326;
const int INFOSET_NUMBER=36;
const double INF=1e18;
const int ACTION_DIM=5;
const int CUDA_NUMBER=6;
const int VALUE_WARM_ITERATOR=20;
const double EPS=1e-20;
const double BIG_BLIND_V=0.01;
const double MIN_BIG_BLINDS=10;
const double MAX_BIG_BLINDS=100;
const double ACTION_NOISE=0.15;//TRAINING
const bool HIDE_NEG_FLAG=true;//evaluation
const int MAX_TREE_SZ=40000;
const int MIN_ITERATOR_NUMBER=100;
const int MAX_ITERATOR_NUMBER=250;
const int EVENT_LIMIT=10;
const int EVENT_DIM=32;//CHANGE
const vector<double>DEFAULT_SCALE={0.5,1.0,2.0,4.0,8.0};
const string critic_dir="../model/";//NEW
const string action_data_dir="../data/";
const int FOLD_PUBLIC_STATE=0;
const int SHOWDOWN_PUBLIC_STATE=1;
const int CHANCE_PUBLIC_STATE=2;
const int PLAYER_PUBLIC_STATE=3;
const int ALLIN_PUBLIC_STATE=4;
const int CFV_PUBLIC_STATE=5;
const int ACTION_TYPE_FOLD=0;
const int ACTION_TYPE_CALL=1;
const int ACTION_TYPE_RAISE=2;
const int ACTION_TYPE_CHANCE=3;
const int PREFLOP_STAGE=0;
const int FLOP_STAGE=1;
const int TURN_STAGE=2;
const int RIVER_STAGE=3;
const int STAGE_NUMBER=1;
const int OOP_PLAYER=0;
const int IP_PLAYER=1;
const int CARD_INDEX_NUMBER=13;
const int CARD_SUIT_NUMBER=4;
const int DEFAULT_ABSTRACTION=0;
const int RLCFR_ABSTRACTION=1;
const int ACTION_FOLD=0;
const int ACTION_CALL=1;
const int ACTION_RAISE=2;
const int ACTION_CHANCE=3;
int THREAD_ID;
int index1[HANDS_NUMBER],index2[HANDS_NUMBER],to_index[CARD_NUMBER][CARD_NUMBER];
int start_id[HANDS_NUMBER];
int compare_ans[INFOSET_NUMBER][INFOSET_NUMBER];
double randvalue(double minv, double maxv){
    int BASE=10000+rand()%9999;
    return minv+(maxv-minv)*(rand()%BASE)/(BASE-1);
}
void updatev(double &x,double y,double r){
    x=(1.0-r)*x+r*y;
}
template<typename T> vector<T> connect_vector(vector<T>vc1, vector<T>vc2){
    vector<T>vc3;
    vc3.insert(vc3.end(),vc1.begin(),vc1.end());
    vc3.insert(vc3.end(),vc2.begin(),vc2.end());
    return vc3;
}
void write_data(vector<double> vec){
    for(double x:vec)printf("%.12lf ",x);
}
void write_data(vector<float>vec){
    for(float x:vec)printf("%.12f ",x);
}
void write_data(vector<vector<float>>vec){
    for(vector<float>vv:vec)
        for(float x:vv)printf("%.12f ",x);
}
bool normalization(vector<double>&v){
    double sum_prob=.0;
    for(int i=0;i<HANDS_NUMBER;i++)sum_prob+=v[i];
    if(sum_prob<EPS)return false;
    sum_prob=1/sum_prob;
    for(int i=0;i<HANDS_NUMBER;i++)v[i]*=sum_prob;
    return true;
}
#endif
