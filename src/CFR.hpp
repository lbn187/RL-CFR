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
#define floord(n,d) floor(((double)(n))/((double)(d)))
torch::jit::script::Module actor_module;
vector<vector<float>>rl_state_data,rl_action_data;
vector<float>rl_value_data;
vector<float>choose_actions;
vector<double>root_actions;
int gpu_id;
normal_distribution<float>distribution(0.0,ACTION_NOISE);
vector<double> get_action(vector<float>state,double noise=0.0){
    default_random_engine generator{(unsigned int)time(0)};
    vector<float>state2=state;
    torch::DeviceType device=at::kCUDA;
    torch::Tensor state_tensor=at::from_blob(state.data(),{1,EVENT_DIM},at::TensorOptions().dtype(at::kFloat));
    std::vector<torch::jit::IValue>inputs;
    inputs.push_back(state_tensor.to(device));
    at::Tensor output;
    try{
        output=actor_module.forward(inputs).toTensor().to(at::kCPU);
    }catch(const c10::Error& e){
        cerr<<"ACTOR NETWORK FORWARD ERROR"<<endl;
    }
    vector<float> v(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
    vector<double>actions;
    for(int i=0;i<6;i++){
        float x,y;//=v[i*2],y=v[i*2+1];
        x=v[i*2],y=v[i*2+1];
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
vector<double> get_target_action(vector<float>state,double noise=0.0){
    default_random_engine generator;
    normal_distribution<float>distribution(0.0,noise);
    torch::DeviceType device=at::kCUDA;
    torch::Tensor state_tensor=at::from_blob(state.data(),{1,EVENT_DIM},at::TensorOptions().dtype(at::kFloat));
    std::vector<torch::jit::IValue>inputs;
    inputs.push_back(state_tensor.to(device));
    at::Tensor output;
    try{
        output=actor_module.forward(inputs).toTensor().to(at::kCPU);
    }catch(const c10::Error& e){
        cerr<<"ACTOR NETWORK FORWARD ERROR"<<endl;
    }
    vector<float> v(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
    vector<double>actions;
    for(int i=0;i<6;i++){
        float x,y;//=v[i*2],y=v[i*2+1];
        x=v[i*2],y=v[i*2+1];
        if(noise>0.000001)x=x+distribution(generator),y=y+distribution(generator);
        x=min(max(x,(float)-1.0),(float)1.0);
        y=min(max(y,(float)-1.0),(float)1.0);
        if(y<0)continue;
        actions.push_back(x);
    }
    return actions;
}
PS get_ps_from_root(PS ps,vector<ACTION>actions){
    for(ACTION a:actions)ps.trans(a);
    return ps;
}
double compare(int myhand,int oppohand,vector<Card>die_cards){
    Card cd1(index1[myhand]),cd2(index2[myhand]),cd3(index1[oppohand]),cd4(index2[oppohand]);
    die_cards.push_back(cd1);
    die_cards.push_back(cd2);
    die_cards.push_back(cd3);
    die_cards.push_back(cd4);
    static bool FLAG[HANDS_NUMBER],VV[CARD_NUMBER];
    for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
    for(Card cd:die_cards)VV[cd.index]=true;
    vector<Card>public_cards;
    for(int i=0;i<CARD_NUMBER;i++)if(!VV[i])public_cards.push_back(Card(i));
    vector<Card>p1cards=public_cards;
    vector<Card>p2cards=public_cards;
    p1cards.push_back(cd1);
    p1cards.push_back(cd2);
    p2cards.push_back(cd3);
    p2cards.push_back(cd4);
    int rk1=get_rank_from_cards(p1cards);
    int rk2=get_rank_from_cards(p2cards);
    if(rk1<rk2)return 1.0;
    if(rk1==rk2)return 0.0;
    return -1.0;
}
//0 BASIC with noise 1 BASIC     2   RL ACTION      3 ALL RL ACTION
double call_value(int lx,int update_next_flag,vector<ACTION>&history_actions,vector<Card>public_cards,double BIG_BLINDS,vector<double>&oop_prob,vector<double>&ip_prob,int oop_action_abstraction,int ip_action_abstraction,vector<ACTION>&final_actions,vector<vector<double>>&total_policy){
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
            if(callv>EPS)actions.push_back(ACTION(0));
            actions.push_back(ACTION(1));
            if(maxraise>potv*0.5){
                if(lx==2){
                    if(h==root_id){
                        for(double scale:root_actions)
                            if(potv*0.5<maxraise)actions.push_back(ACTION(2,potv*0.5+potv*((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                    }else{
                         for(double scale:DEFAULT_SCALE)
                            if(potv*scale<maxraise)actions.push_back(ACTION(2,potv*scale));
                    }
                }else if(lx==3){
                    if(h==root_id){
                        for(double scale:root_actions)
                            if(potv*0.5<maxraise)actions.push_back(ACTION(2,potv*0.5+potv*((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                    }else{
                        vector<double>raise_actions=get_target_action(node.ps.ps_vector());
                        for(double scale:raise_actions)
                            if(potv*0.5<maxraise)actions.push_back(ACTION(2,potv*0.5+potv*((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                    }
                }else if(lx==1){
                    if(node.ps.player==OOP_PLAYER){
                        if(oop_action_abstraction==0){
                            for(double scale:DEFAULT_SCALE)
                                if(potv*scale<maxraise)actions.push_back(ACTION(2,potv*scale));
                        }else{
                            vector<double>raise_actions=get_target_action(node.ps.ps_vector());
                            for(double scale:raise_actions)
                                if(potv*0.5<maxraise)actions.push_back(ACTION(2,potv*0.5+potv*((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                        }
                    }else{
                        if(ip_action_abstraction==0){
                            for(double scale:DEFAULT_SCALE)
                                if(potv*scale<maxraise)actions.push_back(ACTION(2,potv*scale));
                        }else{
                            vector<double>raise_actions=get_target_action(node.ps.ps_vector());
                            for(double scale:raise_actions)
                                if(potv*0.5<maxraise)actions.push_back(ACTION(2,potv*0.5+potv*((scale+1.0)/2.0)*(maxraise-potv*0.5)));
                        }
                    }
                }
                actions.push_back(ACTION(2,maxraise));
                if(h==r_id)if(haite+1<=(int)history_actions.size()){
                    actions.push_back(history_actions[haite++]),r_id=t+(int)actions.size()-1;
                    if(haite==(int)history_actions.size())root_id=r_id;
                }
            }
            if(h==root_id)final_actions=actions;
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
    int DIFF_RANKS=0;
    int hand_cnt=0,card_cnt=0;;
    int TREE_SZ=t;
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
    for(int T=0;T<real_iterator_number;T++){//L4
	    int up_player=T%2;
        for(int node=1;node<TREE_SZ;node++){//L5
            int player=1-tree[node].ps.player;
            for(int i=0;i<hand_cnt;i++){//L6
                if(regret_plus_sum[tree[node].fa][i]<EPS)policy[node][i]=1.0/(tree[tree[node].fa].child_end-tree[tree[node].fa].child_begin+1);
                else policy[node][i]=max(regret_plus[node][i],.0)/regret_plus_sum[tree[node].fa][i];
                hand_prob[player][node][i]=hand_prob[player][tree[node].fa][i]*policy[node][i];
                hand_prob[1-player][node][i]=hand_prob[1-player][tree[node].fa][i];
            }
        }//R5
        int ite=0;
        for(int node=0;node<TREE_SZ;node++){//L5
            int player=tree[node].ps.player;
            if(tree[node].ps.type==FOLD_PUBLIC_STATE){//L6
                for(int pl=0;pl<2;pl++){//L7
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
                }//R7
            }//R6
            if(tree[node].ps.type==SHOWDOWN_PUBLIC_STATE){
                for(int pl=0;pl<2;pl++){
                    for(int i=0;i<hand_cnt;i++){
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        hand_value[pl][node][i]=0.0;
                        for(int j=0;j<hand_cnt;j++){
                            int j1=index1[handid[j]],j2=index2[handid[j]];
                            if(i1!=j1&&i1!=j2&&i2!=j1&&i2!=j2){
                                hand_value[pl][node][i]+=tree[node].ps.totalv/2*compare_ans[i][j]*hand_prob[1-pl][node][j];
                            }
                        }
                    }
                }
            }
            if(tree[node].ps.type==PLAYER_PUBLIC_STATE){//L6
                for(int i=0;i<hand_cnt;i++){
                    hand_value[0][node][i]=hand_value[1][node][i]=0.0;
                    if(tree[node].ps.player==up_player)regret_plus_sum[node][i]=0.0;
                }
            }//R6
        }//R5
        for(int node=TREE_SZ-1;node;node--){//L5
            int player=1-tree[node].ps.player;
            for(int i=0;i<hand_cnt;i++){//L6
                hand_value[player][tree[node].fa][i]+=hand_value[player][node][i]*policy[node][i];
                hand_value[1-player][tree[node].fa][i]+=hand_value[1-player][node][i];
            }//R6
        }//R5
        double pos_factor=1.0*(T+1)*sqrt(T+1)/(1.0*(T+1)*sqrt(T+1)+1),neg_factor=0.5;
        for(int node=TREE_SZ-1;node;node--)if(1-tree[node].ps.player==up_player){
            for(int i=0;i<hand_cnt;i++){//L7
                if(regret_plus[node][i]>=0)regret_plus[node][i]*=pos_factor;else regret_plus[node][i]*=neg_factor;//DCFR
                regret_plus[node][i]+=hand_value[up_player][node][i]-hand_value[up_player][tree[node].fa][i];
                regret_plus_sum[tree[node].fa][i]+=max(regret_plus[node][i],.0);
            }//R7
        }
        if(T>=VALUE_WARM_ITERATOR){
            for(int node=0;node<TREE_SZ;node++)
                for(int i=0;i<hand_cnt;i++){//L6
                    updatev(average_value[0][node][i],hand_value[0][node][i],2.0/(T-VALUE_WARM_ITERATOR+2));
                    updatev(average_value[1][node][i],hand_value[1][node][i],2.0/(T-VALUE_WARM_ITERATOR+2));
                }//R6
            for(int node=0;node<TREE_SZ;node++)
                for(int i=0;i<hand_cnt;i++)
                    updatev(average_policy[node][i],policy[node][i],2.0/(T-VALUE_WARM_ITERATOR+2));
        }

    }//R4
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
    if(update_next_flag){
        vector<double>probs;double sum_prob=0.0;vector<int>ton;
        int act_num=tree[root_id].child_end-tree[root_id].child_begin+1;
        for(int node=tree[root_id].child_begin;node<=tree[root_id].child_end;node++){
            double prob=0.0;
            for(int i=0;i<hand_cnt;i++)
                if(tree[root_id].ps.player==0)prob+=average_policy[node][i]*oop_prob[handid[i]];
                else prob+=average_policy[node][i]*ip_prob[handid[i]];
            if(tree[node].ps.type==PLAYER_PUBLIC_STATE){
                probs.push_back(prob),sum_prob+=prob,ton.push_back(node);
            }
        }
        int nxt_id;
        if(rand()%4==0||sum_prob<EPS){
            nxt_id=tree[root_id].child_begin+rand()%act_num;
        }else{
            double x=randvalue(0.0,sum_prob),y=0.0;
            nxt_id=tree[root_id].child_end;
            for(int i=0;i<(int)probs.size();i++)if(y+probs[i]>x){
                nxt_id=ton[i];
                break;
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
    return value;
}
void add_sav_data(vector<float>state,vector<float>choose_actions,double base_value,double action_value){
    rl_state_data.push_back(state);
    rl_action_data.push_back(choose_actions);
    rl_value_data.push_back((action_value-base_value)*100000);
}
void train_with_action(double big_blinds,bool trust_action){
    get_hand_id();
    vector<vector<double>>policy;
    vector<ACTION>actions;
    PS ps;
    vector<Card>public_cards;
    for(int i=0;i<43;i++){
        while(true){
            int x=rand()%CARD_NUMBER;
            Card cd(x);
            bool flag=true;
            for(Card p:public_cards)if(cd==p){
                flag=false;
                break;
            }
            if(flag){
                public_cards.push_back(cd);
                break;
            }
        }
    }
    vector<double>oop_prob,ip_prob;
    vector<ACTION>history_actions;
    ps.reset(big_blinds,public_cards);
    static bool FLAG[HANDS_NUMBER],VV[CARD_NUMBER];
    int cnt=0;
    for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
    for(Card cd:public_cards)VV[cd.index]=true;
    for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])FLAG[i]=false;else FLAG[i]=true,cnt++;
    oop_prob.resize(HANDS_NUMBER,0.0);
    ip_prob.resize(HANDS_NUMBER,0.0);
    vector<int>hand_id;
    for(int i=0;i<HANDS_NUMBER;i++)if(FLAG[i]){
        oop_prob[i]=1.0/cnt,ip_prob[i]=1.0/cnt;
        hand_id.push_back(i);
    }
    for(int i=0;i<(int)hand_id.size();i++)
        for(int j=0;j<(int)hand_id.size();j++)
            compare_ans[i][j]=compare(hand_id[i],hand_id[j],public_cards);
    while(ps.type==3){
        if(ps.maxraise<EPS)break;
        if(ps.type==3){
            choose_actions.clear();
            root_actions=get_action(ps.ps_vector(),ACTION_NOISE);
            double base_value,action_value;
            if(rand()%3>0){
                base_value=call_value(1,0,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0,actions,policy);
                action_value=call_value(trust_action?3:2,1,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0,actions,policy);
            }else{
                action_value=call_value(trust_action?3:2,0,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0,actions,policy);
                base_value=call_value(1,1,history_actions,public_cards,big_blinds,oop_prob,ip_prob,0,0,actions,policy);
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
    srand(time(0)+thread_id*3333);
}
void training_with_action(string action_data_dir,int epoch,int thread_id,int sample_number,int trust_action=0){
    pre_setting(thread_id);
    for(int i=0;i<sample_number;i++)
        train_with_action(randvalue(5.0,150.0),trust_action);
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
#endif
