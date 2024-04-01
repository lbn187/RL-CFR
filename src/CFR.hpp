#ifndef CFR_H
#define CFR_H
#include"utils.hpp"
#include"ACTION.hpp"
#include"Card.hpp"
#include"PS.hpp"
#include"TreeNode.hpp"
#include<bits/stdc++.h>
#include <memory>
#include<ctime>
#include <torch/script.h>
#include <torch/torch.h>
#include<ATen/ATen.h>
#include <cuda_runtime_api.h>
torch::jit::script::Module actor_module;
torch::jit::script::Module critic_module;
vector<vector<float>>rl_state_data,rl_action_data;
vector<float>rl_value_data;
vector<float>choose_actions;
vector<double>root_actions;
int gpu_id;
normal_distribution<float>distribution(0.0,ACTION_NOISE);
vector<double> get_action(vector<float>state,double noise=0.0){
    default_random_engine generator{(unsigned int)time(0)};
    torch::DeviceType device=at::kCUDA;
    torch::Tensor state_tensor=at::from_blob(state.data(),{1,EVENT_DIM},at::TensorOptions().dtype(at::kFloat));
    std::vector<torch::jit::IValue>inputs;
    inputs.push_back(state_tensor.to(device));
    at::Tensor output;
    output=actor_module.forward(inputs).toTensor().to(at::kCPU);
    vector<float> v(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
    vector<double>actions;
    for(int i=0;i<ACTION_DIM;i++){
        float x=v[i*2],y=v[i*2+1];
        if(noise>0.000001)x=x+distribution(generator),y=y+distribution(generator);
        x=min(max(x,(float)-1.0),(float)1.0);
        y=min(max(y,(float)-1.0),(float)1.0);
        choose_actions.push_back(x);
        choose_actions.push_back(y);
        if(y<0)continue;
        actions.push_back(x);
    }
    return actions;
}
vector<double> get_target_action(vector<float>state,double maxraise,double noise=0.0){
    default_random_engine generator;
    vector<float>state2=state;
    normal_distribution<float>distribution(0.0,noise);
    torch::DeviceType device=at::kCUDA;
    torch::Tensor state_tensor=at::from_blob(state.data(),{1,EVENT_DIM},at::TensorOptions().dtype(at::kFloat));
    std::vector<torch::jit::IValue>inputs;
    inputs.push_back(state_tensor.to(device));
    at::Tensor output;
    output=actor_module.forward(inputs).toTensor().to(at::kCPU);
    vector<float> v(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
    vector<double>actions;
    for(int i=0;i<ACTION_DIM;i++){
        float x=v[i*2],y=v[i*2+1];
        if(noise>0.000001)x=x+distribution(generator),y=y+distribution(generator);
        state2.push_back(x);
        state2.push_back(y);
        x=min(max(x,(float)-1.0),(float)1.0);
        y=min(max(y,(float)-1.0),(float)1.0);
        if(y<0)continue;
        actions.push_back(x);
    }
    if(HIDE_NEG_FLAG){
        torch::Tensor state_tensor2=at::from_blob(state2.data(),{1,EVENT_DIM+ACTION_DIM*2},at::TensorOptions().dtype(at::kFloat));
        std::vector<torch::jit::IValue>inputs2;
        inputs2.push_back(state_tensor2.to(device));
        at::Tensor output;
        output=critic_module.forward(inputs2).toTensor().to(at::kCPU);
        vector<float> v2(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
        if(v2[0]<0)return {-INF};
    }
    return actions;
}
double call_value(int lx,int update_next_flag,vector<ACTION>&history_actions,vector<Card>public_cards,double BIG_BLINDS,vector<double>&oop_prob,vector<double>&ip_prob,int oop_action_abstraction,int ip_action_abstraction,int root_oop_hand_id=-1,int root_ip_hand_id=-1
){
    PS ps;
    ps.reset(BIG_BLINDS,public_cards);
    vector<TreeNode>tree;
    int h=0,t=1,haite=0,r_id=0,root_id=-1;
    if((int)history_actions.size()==0)root_id=0,r_id=-1;
    tree.push_back(TreeNode(ps,0,1,-1,ps.get_final_action()));
    for(;h<t;h++){
        TreeNode node=tree[h];
        if(node.ps.type==PLAYER_PUBLIC_STATE){
            vector<ACTION>actions;
            double totalv=node.ps.totalv,callv=node.ps.callv,maxraise=node.ps.maxraise,potv=totalv+callv,player=node.ps.player;
            if(callv>EPS)actions.push_back(ACTION(ACTION_FOLD));
            actions.push_back(ACTION(ACTION_CALL));
            if(maxraise>potv*0.5){
                if(lx==2){
                    if(h==root_id){
                        for(double scale:root_actions)
                            if(potv*0.5<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*0.5+((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                    }else{
                         for(double scale:DEFAULT_SCALE)
                            if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                    }
                }else if(lx==3){
                    if(h==root_id){
                        for(double scale:root_actions)
                            if(potv*0.5<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*0.5+((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                    }else{
                        vector<double>raise_actions=get_target_action(node.ps.ps_vector(),ps.maxraise);
                        if((int)raise_actions.size()==1&&raise_actions[0]<-1e5){
                            for(double scale:DEFAULT_SCALE)
                                if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                        }else{
                            for(double scale:raise_actions)
                                if(potv*0.5<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*0.5+((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                        }
                    }
                }else if(lx==4){
                    if(h==root_id){
                        for(double scale:DEFAULT_SCALE)
                            if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                    }else{
                        vector<double>raise_actions=get_target_action(node.ps.ps_vector(),ps.maxraise);
                        if((int)raise_actions.size()==1&&raise_actions[0]<-1e5){
                            for(double scale:DEFAULT_SCALE)
                                if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                        }else{
                            for(double scale:raise_actions)
                                if(potv*0.5<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*0.5+((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                        }
                    }
                }else if(lx==1){
                    if(node.ps.player==OOP_PLAYER){
                        if(oop_action_abstraction==DEFAULT_ABSTRACTION){
                            for(double scale:DEFAULT_SCALE)
                                if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                        }else{
                            vector<double>raise_actions=get_target_action(node.ps.ps_vector(),ps.maxraise);
                            if((int)raise_actions.size()==1&&raise_actions[0]<-1e5){
                                for(double scale:DEFAULT_SCALE)
                                    if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                            }else{
                                for(double scale:raise_actions)
                                    if(potv*0.5<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*0.5+((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                            }
                        }
                    }else{
                        if(ip_action_abstraction==DEFAULT_ABSTRACTION){
                            for(double scale:DEFAULT_SCALE)
                                if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                        }else{
                            vector<double>raise_actions=get_target_action(node.ps.ps_vector(),ps.maxraise);
                            if((int)raise_actions.size()==1&&raise_actions[0]<-1e5){
                                for(double scale:DEFAULT_SCALE)
                                    if(potv*scale<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*scale));
                            }else{
                                for(double scale:raise_actions)
                                    if(potv*0.5<maxraise)actions.push_back(ACTION(ACTION_RAISE,potv*0.5+((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                            }
                        }
                    }
                }
                actions.push_back(ACTION(ACTION_RAISE,maxraise));
                if(h==r_id)if(haite+1<=(int)history_actions.size()){
                    actions.push_back(history_actions[haite++]),r_id=t+(int)actions.size()-1;
                    if(haite==(int)history_actions.size())root_id=r_id;
                }
            }
            tree[h].child_begin=t;
            for(ACTION a:actions){
                PS ps=tree[h].ps;
                ps.trans(a);
                TreeNode sonnode(ps,t,tree[h].depth+1,h,a);
                tree.push_back(sonnode);
                t++;
            }
            tree[h].child_end=t-1;
        }
    }
    if(root_id==-1)cerr<<"ROOT ID ERROR"<<endl;
    static bool FLAG[HANDS_NUMBER],VV[CARD_NUMBER];
    int hand_cnt=0,card_cnt=0,TREE_SZ=t;
    if(TREE_SZ>=MAX_TREE_SZ)cerr<<"TREE SIZE BLOOM"<<endl;
    static double hand_prob[2][MAX_TREE_SZ][INFOSET_NUMBER],hand_value[2][MAX_TREE_SZ][INFOSET_NUMBER],regret_plus[MAX_TREE_SZ][INFOSET_NUMBER],policy[MAX_TREE_SZ][INFOSET_NUMBER],regret_plus_sum[MAX_TREE_SZ][INFOSET_NUMBER];
    static double average_hand_prob[2][MAX_TREE_SZ][INFOSET_NUMBER],average_policy[MAX_TREE_SZ][INFOSET_NUMBER],average_value[2][MAX_TREE_SZ][INFOSET_NUMBER];
    static int cardid[CARD_NUMBER],handid[INFOSET_NUMBER];
    for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
    for(Card cd:ps.public_cards)VV[cd.index]=true;
    for(int i=0;i<CARD_NUMBER;i++)if(!VV[i])cardid[card_cnt]++;
    for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])FLAG[i]=false;else FLAG[i]=true,handid[hand_cnt++]=i;
    for(int i=0;i<hand_cnt;i++)hand_prob[0][0][i]=hand_prob[1][0][i]=average_hand_prob[0][0][i]=average_hand_prob[1][0][i]=1.0/hand_cnt;
    for(int node=0;node<TREE_SZ;node++)
        for(int i=0;i<hand_cnt;i++){
            regret_plus_sum[node][i]=0.0;
            regret_plus[node][i]=0.0;
        }
    int real_iterator_number=MAX_ITERATOR_NUMBER;
    for(int T=0;T<real_iterator_number;T++){
	    int up_player=T%2;
        for(int node=1;node<TREE_SZ;node++){
            int player=1-tree[node].ps.player;
            for(int i=0;i<hand_cnt;i++){
                if(regret_plus_sum[tree[node].fa][i]<EPS)policy[node][i]=1.0/(tree[tree[node].fa].child_end-tree[tree[node].fa].child_begin+1);
                else policy[node][i]=max(regret_plus[node][i],.0)/regret_plus_sum[tree[node].fa][i];
                hand_prob[player][node][i]=hand_prob[player][tree[node].fa][i]*policy[node][i];
                hand_prob[1-player][node][i]=hand_prob[1-player][tree[node].fa][i];
            }
        }
        int ite=0;
        for(int node=0;node<TREE_SZ;node++){
            int player=tree[node].ps.player;
            if(tree[node].ps.type==FOLD_PUBLIC_STATE){
                for(int pl=0;pl<2;pl++){
                    vector<double>sum_prob(CARD_NUMBER,.0);
                    double all_sum_prob=.0;
                    for(int i=0;i<hand_cnt;i++){
                        double other_prob=0.0;
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        double x=hand_prob[1-pl][node][i];
                        sum_prob[i1]+=x;
                        sum_prob[i2]+=x;
                        all_sum_prob+=x;
                    }
                    for(int i=0;i<hand_cnt;i++){
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        double other_prob=all_sum_prob-sum_prob[i1]-sum_prob[i2]+hand_prob[1-pl][node][i];
                        if(pl==player)hand_value[pl][node][i]=tree[node].ps.totalv/2*other_prob;
                        else hand_value[pl][node][i]=-tree[node].ps.totalv/2*other_prob;
                    }
                }
            }
            if(tree[node].ps.type==SHOWDOWN_PUBLIC_STATE){
                for(int pl=0;pl<2;pl++){
                    for(int i=0;i<hand_cnt;i++){
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        hand_value[pl][node][i]=0.0;
                        for(int j=0;j<hand_cnt;j++){
                            int j1=index1[handid[j]],j2=index2[handid[j]];
                            if(i1!=j1&&i1!=j2&&i2!=j1&&i2!=j2)
                                hand_value[pl][node][i]+=tree[node].ps.totalv/2*compare_ans[i][j]*hand_prob[1-pl][node][j];
                        }
                    }
                }
            }
            if(tree[node].ps.type==PLAYER_PUBLIC_STATE){
                for(int i=0;i<hand_cnt;i++){
                    hand_value[0][node][i]=hand_value[1][node][i]=0.0;
                    if(tree[node].ps.player==up_player)regret_plus_sum[node][i]=0.0;
                }
            }
        }
        for(int node=TREE_SZ-1;node;node--){
            int player=1-tree[node].ps.player;
            for(int i=0;i<hand_cnt;i++){
                hand_value[player][tree[node].fa][i]+=hand_value[player][node][i]*policy[node][i];
                hand_value[1-player][tree[node].fa][i]+=hand_value[1-player][node][i];
            }
        }
        double pos_factor=1.0*(T+1)*sqrt(T+1)/(1.0*(T+1)*sqrt(T+1)+1),neg_factor=0.5;
        for(int node=TREE_SZ-1;node;node--)if(1-tree[node].ps.player==up_player){
            for(int i=0;i<hand_cnt;i++){
                if(regret_plus[node][i]>=0)regret_plus[node][i]*=pos_factor;else regret_plus[node][i]*=neg_factor;
                regret_plus[node][i]+=hand_value[up_player][node][i]-hand_value[up_player][tree[node].fa][i];
                regret_plus_sum[tree[node].fa][i]+=max(regret_plus[node][i],.0);
            }
        }
        if(T>=VALUE_WARM_ITERATOR){
            for(int node=0;node<TREE_SZ;node++)
                for(int i=0;i<hand_cnt;i++){
                    updatev(average_value[0][node][i],hand_value[0][node][i],2.0/(T-VALUE_WARM_ITERATOR+2));
                    updatev(average_value[1][node][i],hand_value[1][node][i],2.0/(T-VALUE_WARM_ITERATOR+2));
                }
            for(int node=0;node<TREE_SZ;node++)
                for(int i=0;i<hand_cnt;i++)
                    updatev(average_policy[node][i],policy[node][i],2.0/(T-VALUE_WARM_ITERATOR+2));
        }
    }
    for(int node=1;node<TREE_SZ;node++){
        int player=1-tree[node].ps.player;
        for(int i=0;i<hand_cnt;i++){
            average_hand_prob[player][node][i]=hand_prob[player][tree[node].fa][i]*average_policy[node][i];
            average_hand_prob[1-player][node][i]=hand_prob[1-player][tree[node].fa][i];
        }
    }
    double value=0.0;
    for(int i=0;i<hand_cnt;i++){
        if(tree[root_id].ps.player==0)value+=average_value[0][root_id][i]*oop_prob[handid[i]];
        else value+=average_value[1][root_id][i]*ip_prob[handid[i]];
    }
    if(update_next_flag==100){
        int root_hand_id;
        if(tree[root_id].ps.player==0)root_hand_id=root_oop_hand_id;else root_hand_id=root_ip_hand_id;
        if(root_hand_id==-1)cerr<<"ROOT HAND ID ERROR"<<endl;
        double x=randvalue(0.0,1.0),y=0.0;
        int nxt_id=tree[root_id].child_end;
        for(int node=tree[root_id].child_begin;node<=tree[root_id].child_end;node++){
            y+=average_policy[node][root_hand_id];
            if(y>x){
                nxt_id=node;
                break;
            }
        }
        history_actions.push_back(tree[nxt_id].final_action);
        double oop_cfv,ip_cfv,exv=0.0;
        if(tree[root_id].ps.player==0)oop_cfv=value,ip_cfv=-value;else oop_cfv=-value,ip_cfv=value;
        for(int pl=0;pl<2;pl++){
            double cfv=average_value[pl][root_id][pl==0?root_oop_hand_id:root_ip_hand_id],other_prob=0.0,tmp;
            int r1=index1[handid[pl==0?root_oop_hand_id:root_ip_hand_id]],r2=index2[handid[pl==0?root_oop_hand_id:root_ip_hand_id]];
            for(int i=0;i<hand_cnt;i++){
                int i1=index1[handid[i]],i2=index2[handid[i]];
                if(i1!=r1&&i1!=r2&&i2!=r1&&i2!=r2)other_prob+=average_hand_prob[1-pl][root_id][i];
            }
            if(other_prob<EPS)return INF;
            if(pl==0)tmp=(cfv-oop_cfv)/other_prob;else tmp=(cfv-ip_cfv)/other_prob;
            if(pl==tree[root_id].ps.player)exv+=tmp;else exv-=tmp;
        }
        return exv;
    }
    if(update_next_flag>0){
        vector<double>probs;double sum_prob=0.0;vector<int>ton;
        int act_num=tree[root_id].child_end-tree[root_id].child_begin+1;
        for(int node=tree[root_id].child_begin;node<=tree[root_id].child_end;node++){
            double prob=0.0;
            for(int i=0;i<hand_cnt;i++)
                if(tree[root_id].ps.player==0)prob+=average_policy[node][i]*oop_prob[handid[i]];
                else prob+=average_policy[node][i]*ip_prob[handid[i]];
            probs.push_back(prob);
            sum_prob+=prob,ton.push_back(node);
        }
        int nxt_id;
        if(rand()%4==0||sum_prob<EPS){
            nxt_id=tree[root_id].child_begin+rand()%act_num;
        }else{
            double x=randvalue(0.0,sum_prob),y=0.0;
            nxt_id=tree[root_id].child_end;
            for(int i=0;i<(int)probs.size();i++){
                y+=probs[i];
                if(y>x){
                    nxt_id=ton[i];
                    break;
                }
            }
        }
        ACTION a=tree[nxt_id].final_action;
        history_actions.push_back(a);
        vector<double>pre_oop_prob=oop_prob;
        vector<double>pre_ip_prob=ip_prob;
        for(int i=0;i<hand_cnt;i++)
            if(tree[root_id].ps.player==0)oop_prob[handid[i]]*=average_policy[nxt_id][i];
            else ip_prob[handid[i]]*=average_policy[nxt_id][i];
        if(!normalization(oop_prob))oop_prob=pre_oop_prob;
        if(!normalization(ip_prob))ip_prob=pre_ip_prob;
    }
    if(update_next_flag<0){//explo test
        static int best_child[MAX_TREE_SZ][INFOSET_NUMBER];
        int explo_player=1-tree[root_id].ps.player;
        for(int node=0;node<TREE_SZ;node++){
            for(int i=0;i<INFOSET_NUMBER;i++){
                hand_value[0][node][i]=hand_value[1][node][i]=0;
                if(tree[node].ps.player==explo_player)
                    best_child[node][i]=-1;
            }
            if(tree[node].ps.type==FOLD_PUBLIC_STATE){
                int pl=explo_player;
                    vector<double>sum_prob(CARD_NUMBER,.0);
                    double all_sum_prob=.0;
                    for(int i=0;i<hand_cnt;i++){
                        double other_prob=0.0;
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        double x=average_hand_prob[1-pl][node][i];
                        sum_prob[i1]+=x;
                        sum_prob[i2]+=x;
                        all_sum_prob+=x;
                    }
                    for(int i=0;i<hand_cnt;i++){
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        double other_prob=all_sum_prob-sum_prob[i1]-sum_prob[i2]+average_hand_prob[1-pl][node][i];
                        if(pl==tree[node].ps.player)hand_value[pl][node][i]=tree[node].ps.totalv/2*other_prob;
                        else hand_value[pl][node][i]=-tree[node].ps.totalv/2*other_prob;
                    }
            }
            if(tree[node].ps.type==SHOWDOWN_PUBLIC_STATE){
                int pl=explo_player;
                    for(int i=0;i<hand_cnt;i++){
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        hand_value[pl][node][i]=0.0;
                        for(int j=0;j<hand_cnt;j++){
                            int j1=index1[handid[j]],j2=index2[handid[j]];
                            if(i1!=j1&&i1!=j2&&i2!=j1&&i2!=j2)
                                hand_value[pl][node][i]+=tree[node].ps.totalv/2*compare_ans[i][j]*average_hand_prob[1-pl][node][j];
                        }
                    }
            }
        }
        for(int node=TREE_SZ-1;node;node--){
            for(int i=0;i<INFOSET_NUMBER;i++)if(1-tree[node].ps.player==explo_player){
                if(hand_value[explo_player][node][i]>hand_value[explo_player][tree[node].fa][i]||best_child[tree[node].fa][i]==-1){
                    best_child[tree[node].fa][i]=node;
                    hand_value[explo_player][tree[node].fa][i]=hand_value[explo_player][node][i];
                }
            }else hand_value[explo_player][tree[node].fa][i]+=hand_value[explo_player][node][i];
        }
        double value_oppo=0.0;
        for(int i=0;i<hand_cnt;i++){
            if(explo_player==0)value_oppo+=hand_value[0][root_id][i]*oop_prob[handid[i]];
            else value_oppo+=hand_value[1][root_id][i]*ip_prob[handid[i]];
        }
        return value_oppo+value;
    }
    return value;
}
void add_sav_data(vector<float>state,vector<float>choose_actions,double base_value,double action_value){
    rl_state_data.push_back(state);
    rl_action_data.push_back(choose_actions);
    rl_value_data.push_back((action_value-base_value)*100000);
}
void train_with_action(double big_blinds,bool trust_action){
    PS ps;
    vector<Card>public_cards=dealt_cards();
    vector<double>oop_prob,ip_prob;
    vector<int>hand_id;
    vector<ACTION>history_actions;
    ps.reset(big_blinds,public_cards);
    set_prob(public_cards,oop_prob,ip_prob,hand_id);
    while(ps.type==3){
        if(ps.maxraise<EPS)break;
        if(ps.type==3){
            choose_actions.clear();
            root_actions=get_action(ps.ps_vector(),ACTION_NOISE);
            double base_value,action_value;
            if(rand()%3>0){
                base_value=call_value(trust_action?4:1,0,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0);
                action_value=call_value(trust_action?3:2,1,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0);
            }else{
                action_value=call_value(trust_action?3:2,0,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0);
                base_value=call_value(trust_action?4:1,1,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0);
            }
            add_sav_data(ps.ps_vector(),choose_actions,base_value,action_value);
            ps.trans(history_actions.back());
        }
    }
}
void pre_setting(int thread_id){
    gpu_id=thread_id%CUDA_NUMBER+1;
    cudaSetDevice(gpu_id);
    torch::DeviceType device=at::kCUDA;
    THREAD_ID=thread_id;
    string actor_dir=critic_dir+"actor.pt";
    try{
        actor_module=torch::jit::load(actor_dir,torch::Device(torch::DeviceType::CUDA,gpu_id));
        actor_module.to(device);
    }
    catch(const c10::Error& e){
        cerr<<"ACTOR  MODULE LOAD ERROR"<<endl;
    }
    if(HIDE_NEG_FLAG){
        try{
            critic_module=torch::jit::load(critic_dir+"critic.pt",torch::Device(torch::DeviceType::CUDA,gpu_id));
            critic_module.to(device);
        }
        catch(const c10::Error& e){
            cerr<<"ACTOR  MODULE LOAD ERROR"<<endl;
        }
    }
    srand(time(0)+thread_id*3333);
    get_hand_id();
}
double get_value(double big_blinds){
    PS ps;
    vector<Card>public_cards=dealt_cards();
    vector<double>oop_prob,ip_prob;
    vector<int>hand_id;
    vector<ACTION>history_actions;
    ps.reset(big_blinds,public_cards);
    set_prob(public_cards,oop_prob,ip_prob,hand_id);
    double value1=call_value(1,0,history_actions,public_cards,big_blinds,oop_prob,ip_prob,1,0);
    double value2=call_value(1,0,history_actions,public_cards,big_blinds,oop_prob,ip_prob,1,1);
    return value1-value2;
}
pair<double,double> AI_vs_AI(int oop_ai,int ip_ai,int hand_id_oop,int hand_id_ip,double big_blinds){
    vector<Card>public_cards=dealt_cards();
    vector<double>oop_prob,ip_prob;
    vector<int>hand_id;
    vector<ACTION>history_actions;
    PS ps;
    ps.reset(big_blinds,public_cards);
    set_prob(public_cards,oop_prob,ip_prob,hand_id);
    double exv=0.0;
    exv=call_value(1,100,history_actions,public_cards,big_blinds,oop_prob,ip_prob,1,1,hand_id_oop,hand_id_ip);
    if(exv>1e10)exv=0.0;
    history_actions.clear();
    ps.reset(big_blinds,public_cards);
    while(ps.type==PLAYER_PUBLIC_STATE){
        call_value(1,100,history_actions,public_cards,big_blinds,oop_prob,ip_prob,oop_ai,ip_ai,hand_id_oop,hand_id_ip);
        //if(ps.player==OOP_PLAYER)call_value(1,100,history_actions,public_cards,big_blinds,oop_prob,ip_prob,oop_ai,oop_ai,hand_id_oop,hand_id_ip);
        //else call_value(1,100,history_actions,public_cards,big_blinds,oop_prob,ip_prob,ip_ai,ip_ai,hand_id_oop,hand_id_ip);
        ps.trans(history_actions.back());
    }
    double truev;
    if(ps.type==FOLD_PUBLIC_STATE){
        if(ps.player==0)truev=-ps.totalv/2;else truev=ps.totalv/2;
    }else
    if(ps.type==SHOWDOWN_PUBLIC_STATE){
        truev=-ps.totalv/2*compare_ans[hand_id_oop][hand_id_ip];
    }else cerr<<"PS FINAL TYPE ERROR"<<endl;
    return make_pair(truev-exv,(truev-exv)*(truev-exv));
}
double get_explo(int a1,int a2,double big_blinds){
    PS ps;
    vector<Card>public_cards=dealt_cards();
    vector<double>oop_prob,ip_prob;
    vector<int>hand_id;
    vector<ACTION>history_actions;
    ps.reset(big_blinds,public_cards);
    set_prob(public_cards,oop_prob,ip_prob,hand_id);
    return call_value(1,-1,history_actions,public_cards,big_blinds,oop_prob,ip_prob,a2,a1);
}
void training_with_action(string action_data_dir,int epoch,int thread_id,int sample_number,int trust_action=0){
    pre_setting(thread_id);
    for(int i=0;i<sample_number;i++)
        train_with_action(randvalue(MIN_BIG_BLINDS,MAX_BIG_BLINDS),trust_action);
    string dir=action_data_dir+"rlstate"+to_string(epoch)+"_"+to_string(THREAD_ID)+".csv";
    FILE* fp=freopen(dir.c_str(),"w",stdout);
    write_data(rl_state_data);
    fclose(stdout);
    fflush(fp);
    dir=action_data_dir+"rlaction"+to_string(epoch)+"_"+to_string(THREAD_ID)+".csv";
    fp=freopen(dir.c_str(),"w",stdout);
    write_data(rl_action_data);
    fclose(stdout);
    fflush(fp);
    dir=action_data_dir+"rlvalue"+to_string(epoch)+"_"+to_string(THREAD_ID)+".csv";
    fp=freopen(dir.c_str(),"w",stdout);
    write_data(rl_value_data);
    fclose(stdout);
    fflush(fp);
    freopen("/dev/tty","w",stdout);
    freopen("/dev/tty","r",stdin);
}
double test_abstractions(int num=100){
    double x=0.0;
    pre_setting(0);
    for(int i=0;i<num;i++)
        x+=get_value(randvalue(MIN_BIG_BLINDS,MAX_BIG_BLINDS));
    return x;
}
double test_exploitability(int abstraction1,int abstraction2,int num=100){//abstraction1 be exploited
    double x=0.0;
    pre_setting(0);
    for(int i=0;i<num;i++)
        x+=get_explo(abstraction1,abstraction2,randvalue(MIN_BIG_BLINDS,MAX_BIG_BLINDS));
    return x;
}
pair<double,double> test_AI(int thread_id,int AI1,int AI2,int num=100){
    double v=0.0,s=0.0;
    pre_setting(thread_id);
    for(int i=0;i<num;i++){
        int hand_id_oop=rand()%INFOSET_NUMBER,hand_id_ip=rand()%INFOSET_NUMBER;
        while(hand_id_oop==hand_id_ip)hand_id_oop=rand()%INFOSET_NUMBER,hand_id_ip=rand()%INFOSET_NUMBER;
        if(i%2==0){
            pair<double,double>uii=AI_vs_AI(AI1,AI2,hand_id_oop,hand_id_ip,randvalue(MIN_BIG_BLINDS,MAX_BIG_BLINDS));
            v-=uii.first;
            s+=uii.second;
        }else{
            pair<double,double>uii=AI_vs_AI(AI2,AI1,hand_id_oop,hand_id_ip,randvalue(MIN_BIG_BLINDS,MAX_BIG_BLINDS));
            v+=uii.first;
            s+=uii.second;
        }
    }
    return make_pair(v,s);
}
#endif