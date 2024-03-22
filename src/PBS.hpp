#ifndef PBS_H
#define PBS_H
#include"utils.hpp"
#include"Card.hpp"
struct PBS{
    vector<double>oop_prob,ip_prob;
    vector<Card>public_cards;
    double totalv,callv,maxraise;
    int stage,player,type;
    void reset(double BIG_BLINDS,vector<Card>cds){
        static bool FLAG[HANDS_NUMBER],VV[CARD_NUMBER];
        int cnt=0;
        for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
        for(Card cd:cds)VV[cd.index]=true;
        for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])FLAG[i]=false;else FLAG[i]=true,cnt++;
        oop_prob.resize(HANDS_NUMBER);
        ip_prob.resize(HANDS_NUMBER);
        for(int i=0;i<HANDS_NUMBER;i++)if(FLAG[i])oop_prob[i]=1.0/cnt,ip_prob[i]=1.0/cnt;
        public_cards.clear();
        totalv=BIG_BLIND_V*1.5;
        callv=BIG_BLIND_V*0.5;
        maxraise=ROUND(BIG_BLIND_V*(BIG_BLINDS-1),ROUND_INDEX);
        stage=0;
        player=1;
        type=3;
    }
    void normalization(){
        static bool VV[CARD_NUMBER];
        for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
        for(Card cd:public_cards)VV[cd.index]=true;
        for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])oop_prob[i]=ip_prob[i]=0;
        double sum_ooppb=.0,sum_ippb=.0;
        for(int i=0;i<HANDS_NUMBER;i++)
            if(VV[index1[i]]||VV[index2[i]])sum_ooppb+=0.0;else sum_ooppb+=oop_prob[i],sum_ippb+=ip_prob[i];
        if(sum_ooppb>EPS){
            sum_ooppb=1.0/sum_ooppb;
            for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])oop_prob[i]=.0;else oop_prob[i]*=sum_ooppb;
        }else{
            int cnt=0;
            for(int i=0;i<HANDS_NUMBER;i++)if(!VV[index1[i]]&&!VV[index2[i]])cnt++;
            for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])oop_prob[i]=.0;else oop_prob[i]=1.0/cnt;
        }
        if(sum_ippb>EPS){
            sum_ippb=1.0/sum_ippb;
            for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])ip_prob[i]=.0;else ip_prob[i]*=sum_ippb;
        }else{
            int cnt=0;
            for(int i=0;i<HANDS_NUMBER;i++)if(!VV[index1[i]]&&!VV[index2[i]])cnt++;
            for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])ip_prob[i]=.0;else ip_prob[i]=1.0/cnt;
        }
    }
    PBS &operator=(const PBS&_pbs){
        if(&_pbs==this)return *this;
        oop_prob=_pbs.oop_prob;
        ip_prob=_pbs.ip_prob;
        public_cards=_pbs.public_cards;
        totalv=_pbs.totalv;
        callv=_pbs.callv;
        maxraise=_pbs.maxraise;
        stage=_pbs.stage;
        player=_pbs.player;
        type=_pbs.type;
        return *this;
    }
    void see(){
        printf("%.12lf %.12lf %.12lf   STAGE%d PLAYER%d TYPE%d CARDS%d\n",totalv,callv,maxraise,stage,player,type,(int)public_cards.size());
    }
    void allsee(){
        printf("%.12lf %.12lf %.12lf   STAGE%d PLAYER%d TYPE%d CARDS%d\n",totalv,callv,maxraise,stage,player,type,(int)public_cards.size());
        printf("PUBLIC CARDS:");for(Card cd:public_cards)cd.output();puts("");
    }
    
};
#endif