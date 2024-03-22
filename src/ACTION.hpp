#ifndef ACTION_H
#define ACTION_H
#include"utils.hpp"
#include"Card.hpp"
struct ACTION{
    int type;
    double raise_v;
    vector<Card> cd;
    ACTION(){}
    ACTION(int _type){
        type=_type;
        if(type==2)raise_v=BIG_BLIND_V;else raise_v=.0;
        if(_type>3||_type<0){
            puts("TYPE ERROR");
        }
    }
    ACTION(int _type, double v){
        type=_type;
        //raise_v=ROUND(v,ROUND_INDEX);
        raise_v=v;
        if(_type==2&&raise_v<EPS){
            puts("ACTION ERROR");
        }
        if(_type>3||_type<0){
            puts("TYPE ERROR");
        }
    }
    ACTION(Card _cd){
        type=3;
        cd={_cd};
    }
    ACTION(vector<Card> _cd){
        type=3;
        cd=_cd;
    }
    ACTION &operator=(const ACTION &_action){
        if(&_action==this)return *this;
        type=_action.type;
        raise_v=ROUND(_action.raise_v,ROUND_INDEX);
        cd=_action.cd;
        return *this;
    }
    bool operator<(const ACTION &_action){
        return raise_v<_action.raise_v;
    }
    bool operator==(const ACTION &_action){
        if(type!=_action.type)return false;
        if(type==2&&fabs(raise_v-_action.raise_v)>1e-7)return false;
        if(type==3&&cd!=_action.cd)return false;
        return true;
    }
};
#endif