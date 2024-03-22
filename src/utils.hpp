#ifndef UTILS_H
#define UTILS_H
#include<bits/stdc++.h>
#include<random>
using namespace std;

const bool USE_ACTION_FLAG=false;
const int RAISE_NUMBER=5;
const int MAX_PUBLIC_CARDS=5;
const int CARD_NUMBER=52;
const int HANDS_NUMBER=1326;
const int INFOSET_NUMBER=55;
//int DIFF_RANKS=0;
const double INF=1e18;
const bool TRUST_ACTION_FLAG=false;
const double START_REGRET=0.01;
const double BASE_VALUE=1.0;
const int DIFF_PUBS=6591;
//const int MAX_ACTION_NUMBER=7;
const int ITERATOR_NUMBER=20000;
const int CUDA_NUMBER=6;
const int WARM_ITERATOR=0;//TODO
const int VALUE_WARM_ITERATOR=16;//TODO
const double EPS=1e-20;
const double BIG_BLIND_V=0.01;
const double DEFAULT_BIG_BLINDS=200;
const int ACTION_DIM=3;
const double ACTION_NOISE=0.00;//TRAINING

//const double TOTAL_V=10000;
bool INDEX_TO_CARD_FLAG=false;
bool HANDS_RANK_FLAG=false;
const int DIFF_SAME_NUMBER=5408;
//int index1_dic[HANDS_NUMBER],index2_dic[HANDS_NUMBER];
//int pair_to_index[CARD_NUMBER][CARD_NUMBER];
//vector<int>ranks(HANDS_NUMBER);
double sum_oop=.0,sum_ip=.0;
const bool random_card_flag=true;
const bool HIDE_NEG_FLAG=false;//evaluation
const int MAX_ACTION_NUMBER=9;
const int RANGE_K=11;
const int MAX_DIFF_RANK=300;
const double RANGE_RATIO[RANGE_K]={0.04,0.08,0.12,0.16,0.19,0.22,0.24,0.24,0.24,0.24,0.23};
const int RANGE_TIMES=1;
const int APPROX_K=100;
//const double MAX_RAISE=4.0;
//const double MAXRAISE_INDEX=2.0;
const int STATE_DIM=2678;
const int MAX_TREE_SZ=10000;
const int MIN_ITERATOR_NUMBER=100;
const int MAX_ITERATOR_NUMBER=500;
const double MAX_RAISE_SIZE=5.0;
const int EVENT_LIMIT=10;
const int EVENT_DIM=32;//CHANGE
const int ROUND_INDEX=10000;
const vector<double>DEFAULT_SCALE={0.5,1.0,2.0,4.0,8.0,15.0};
const vector<double>DEFAULT_RAISING={0.25,0.5,0.75,1.0,1.5,2.0};
const vector<double>OPPONENT_SCALE={0.25,0.5,1.0,2.0};
const string critic_dir="../new_model/";//NEW
const string action_data_dir="../data/action_data/";
const bool USE_PBS_ASSUMPTION=true;
const double BAD_ACTION_EPSON=0.01;
const int LIMIT_LOOK_AHEAD_STEP=3;
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
const int DEFAULT_ABSTRACTION_ADD_RANDOM=0;
const int DEFAULT_ABSTRACTION=1;
const int FLOP_NUMBER=22100;
const int PREFLOP_NUMBER=169;
const int AGG_NUMBER=200;
const int CFV_NETWORK_NUMBER=10;
const double CFV_EPSON=0.0001;
const int K_MEANS_ITER=1000;
int THREAD_ID;
int index1[HANDS_NUMBER];
int index2[HANDS_NUMBER];
int to_index[CARD_NUMBER][CARD_NUMBER];
int start_id[HANDS_NUMBER];
struct buffer_info{
    void *ptr;
    ssize_t itemsize;
    std::string format;
    ssize_t ndim;
    std::vector<ssize_t> shape;
    std::vector<ssize_t> strides;
};
//vector<int>batch_ids;
int thread_id;
double randvalue(double minv, double maxv){
    //std::random_device e;
    //std::uniform_real_distribution<double> u(minv,maxv);
    //return u(e);
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
double ROUND(double x,int z){
    return 1.0*(round(x*z))/z;
}
//vector<vector<double>>data_state;
//vector<vector<double>>data_value;

#endif
