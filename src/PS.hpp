#ifndef PS_H
#define PS_H
#include"utils.hpp"
#include"Card.hpp"
#include"ACTION.hpp"
struct PS{
    vector<Card>public_cards;
    double totalv,callv,maxraise;
    int stage,player,type;
    vector<double>events[STAGE_NUMBER];
    vector<ACTION>history_actions;
    vector<float> ps_vector(){
        vector<float>state;
        state.push_back((float)1.0*stage);
        state.push_back((float)1.0*player);
        state.push_back((float)totalv);
        state.push_back((float)callv);
        state.push_back((float)maxraise);
        vector<float>card_index(CARD_INDEX_NUMBER, .0);
        vector<float>card_suit(CARD_SUIT_NUMBER ,.0);
        for(Card cd:public_cards){//L4
            card_index[cd.number()]+=1.0;
            card_suit[cd.suit()]+=1.0;
        }//R4
        for(int i=0;i<CARD_INDEX_NUMBER;i++)state.push_back(card_index[i]);
        for(int i=0;i<CARD_SUIT_NUMBER;i++)state.push_back(card_suit[i]);
        for(int stg=0;stg<STAGE_NUMBER;stg++){
            int id=0;
            for(double x:events[stg]){
                if(id<EVENT_LIMIT)state.push_back(x);
                id++;
            }
            for(int i=(int)events[stg].size();i<EVENT_LIMIT;i++)state.push_back(-1.0);
        }
        while((int)state.size()<EVENT_DIM)state.push_back(-1.0);
        if((int)state.size()>EVENT_DIM){
            cerr<<"EVENT DIM ERROR"<<endl;
            for(;;);
        }
        return state;
    }
    void trans(ACTION a){
        if(a.type==ACTION_TYPE_FOLD){
            totalv-=callv;
            callv=0.0;
            type=FOLD_PUBLIC_STATE;
            events[stage].push_back(-1.0);
        }else if(a.type==ACTION_TYPE_RAISE){
            events[stage].push_back(a.raise_v/(totalv+callv));
            totalv+=callv+a.raise_v;
            callv=a.raise_v;
            maxraise-=a.raise_v;
            type=PLAYER_PUBLIC_STATE;
        }else if(a.type==ACTION_TYPE_CALL){
            events[stage].push_back(0.0);
            if((stage==PREFLOP_STAGE&&totalv>BIG_BLIND_V*1.7)||(stage!=PREFLOP_STAGE&&player==IP_PLAYER)||(stage!=PREFLOP_STAGE&&callv>EPS)||(callv<EPS&&maxraise<EPS)){//L7
                if(stage==RIVER_STAGE)type=SHOWDOWN_PUBLIC_STATE;else type=CHANCE_PUBLIC_STATE;
            }else{
                type=PLAYER_PUBLIC_STATE;
            }//R7
            if(maxraise<EPS&&type!=SHOWDOWN_PUBLIC_STATE&&type!=CHANCE_PUBLIC_STATE)type=ALLIN_PUBLIC_STATE;
            totalv+=callv;
            callv=0.0;
        }else if(a.type==ACTION_TYPE_CHANCE){
            for(Card cd:a.cd)public_cards.push_back(cd);
        }
        history_actions.push_back(a);
    }
    void reset(double BIG_BLINDS,vector<double>cds){
        public_cards.clear();
        totalv=BIG_BLIND_V*1.5;
        callv=BIG_BLIND_V*0.5;
        maxraise=ROUND(BIG_BLIND_V*(BIG_BLINDS-1),ROUND_INDEX);
        stage=PREFLOP_STAGE;
        player=IP_PLAYER;
        type=PLAYER_PUBLIC_STATE;
        for(int i=0;i<STAGE_NUMBER;i++)events[i].clear();
        history_actions.clear();
        public_cards=cds;
    }
    PS &operator=(const PS&_ps){
        if(&_ps==this)return *this;
        public_cards=_ps.public_cards;
        totalv=_ps.totalv;
        callv=_ps.callv;
        maxraise=_ps.maxraise;
        stage=_ps.stage;
        player=_ps.player;
        type=_ps.type;
        for(int i=0;i<STAGE_NUMBER;i++)events[i]=_ps.events[i];
        history_actions=_ps.history_actions;
        return *this;
    }
    void see(){
        printf("%.12lf %.12lf %.12lf   STAGE%d PLAYER%d TYPE%d CARDS%d\n",totalv,callv,maxraise,stage,player,type,(int)public_cards.size());
        printf("PUBLIC CARDS:");for(Card cd:public_cards)cd.output();puts("");
    }
    ACTION get_final_action(){
        if((int)history_actions.size()>0)return history_actions.back();
        return ACTION(3);
    }
};
#endif