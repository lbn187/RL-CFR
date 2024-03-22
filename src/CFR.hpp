#ifndef CFR_H
#define CFR_H
#include"utils.hpp"
#include"ACTION.hpp"
#include"Card.hpp"
#include"PBS.hpp"
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
//#include<pybind11/embed.h>
//#include<pybind11/pybind11.h>
//#include<pybind11/numpy.h>
//#include <pybind11/stl.h>
//namespace py=pybind11;
torch::jit::script::Module module[6];
torch::jit::script::Module actor_module;
torch::jit::script::Module critic_module;
vector<vector<float>>rl_state_data,rl_action_data;
vector<float>rl_value_data;
vector<float>choose_actions;
vector<double>root_actions;
int gpu_id;
bool SEE_ACTION_FLAG=false;
normal_distribution<float>distribution(0.0,ACTION_NOISE);
vector<double> get_action(vector<float>state,double noise=0.0){
    if(maxraise<EPS)return {};
    default_random_engine generator{(unsigned int)time(0)};
    //normal_distribution<float>distribution(0.0,noise);
    vector<float>state2=state;
    torch::DeviceType device=at::kCUDA;
    torch::Tensor state_tensor=at::from_blob(state.data(),{1,EVENT_DIM},at::TensorOptions().dtype(at::kFloat));
    std::vector<torch::jit::IValue>inputs;
    inputs.push_back(state_tensor.to(device));
    /*try{
        actor_module.to(device);
    }catch(const c10::Error& e){
        cerr<<"ACTOR MODULE TO ERROR"<<endl;
    }*/
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
        choose_actions.push_back(x);
        choose_actions.push_back(y);
        if(noise>0.000001)x=x+distribution(generator),y=y+distribution(generator);
        x=min(max(x,(float)-1.0),(float)1.0);
        y=min(max(y,(float)-1.0),(float)1.0);
        if(y<0)continue;
        actions.push_back(x);
    }
    return actions;
}
vector<double> get_target_action(vector<float>state,double noise=0.0){
    //TODO
    if(maxraise<EPS)return {};
    default_random_engine generator;
    normal_distribution<float>distribution(0.0,noise);
    torch::DeviceType device=at::kCUDA;
    torch::Tensor state_tensor=at::from_blob(state.data(),{1,EVENT_DIM},at::TensorOptions().dtype(at::kFloat));
    std::vector<torch::jit::IValue>inputs;
    inputs.push_back(state_tensor.to(device));
    /*try{
        actor_module.to(device);
    }catch(const c10::Error& e){
        cerr<<"ACTOR MODULE TO ERROR3"<<endl;
    }*/
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
    for(Card cd:pbs.public_cards)VV[cd.index]=true;
    vector<Card>public_cards;
    for(int i=0;i<CARD_NUMBER;i++)if(!VV[i])public_cards.push_back(Card(i));
    vector<Card>p1cards=public_cards;
    vector<Card>p2cards=public_cards;
    p1cards.push_back(cd1);
    p1cards.push_back(cd2);
    p2cards.push_back(cd1);
    p2cards.push_back(cd2);
    int rk1=get_rank_from_cards(p1cards);
    int rk2=get_rank_from_cards(p2cards);
    if(rk1<rk2)return 1.0;
    if(rk1==rk2)return 0.0;
    return -1.0;
}
//lx 0-BASIC ACTION WITH NOISE 1-BASIC ACTION 2-RL ACTION 3-ALL RL ACTION
//update_next_flag 0-no update 1-update and save data 2-normal update 3-action step update 4-average policy update
double call_value(int lx=2,//0 BASIC with noise 1 BASIC     2   RL ACTION      3 ALL RL ACTION
int update_next_flag,
vector<ACTION>history_actions,
vector<Card>public_cards,
double BIG_BLINDS,
vector<double> oop_prob,
vector<double> ip_prob,
int oop_action_abstraction,
int ip_action_abstraction,
vector<ACTION>&final_actions,vector<vector<double>>&total_policy){
    //PBS pbs;
    PS ps;
    //pbs.reset(BIG_BLINDS,public_cards);
    ps.reset(BIG_BLINDS,public_cards);
    vector<TreeNode>tree;
    int h=0,t=1,root_id=0,haite=0;
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
                if(haite+1<=(int)history_actions.size())actions.push_back(ACTION(2,history_actions[haite++]));else root_id=h;
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
    static bool FLAG[HANDS_NUMBER],VV[CARD_NUMBER];
    static int ranks[HANDS_NUMBER];
    static double sum_prob[MAX_DIFF_RANK],sum_probi[CARD_NUMBER][MAX_DIFF_RANK],average_value[HANDS_NUMBER*2];
    int DIFF_RANKS=0;
    int hand_cnt=0,card_cnt=0;;
    int TREE_SZ=t;
    if(TREE_SZ>=MAX_TREE_SZ)cerr<<"TREE SIZE BLOOM"<<endl;
    static double hand_prob[2][MAX_TREE_SZ][INFOSET_NUMBER],hand_value[2][MAX_TREE_SZ][INFOSET_NUMBER],regret_plus[MAX_TREE_SZ][INFOSET_NUMBER],policy[MAX_TREE_SZ][INFOSET_NUMBER],regret_plus_sum[MAX_TREE_SZ][INFOSET_NUMBER];
    static double oop_sum[MAX_TREE_SZ], ip_sum[MAX_TREE_SZ];
    static double average_hand_prob[2][MAX_TREE_SZ][INFOSET_NUMBER],average_policy[MAX_TREE_SZ][INFOSET_NUMBER],card_prob[2][CARD_NUMBER];
    static int cardid[CARDS_NUMBER],handid[INFOSET_NUMBER];

    for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
    for(Card cd:pbs.public_cards)VV[cd.index]=true;
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
                if(regret_plus_sum[tree[node].fa][i]<EPS)policy[node][i]=1.0/(tree[node].child_end-tree[node].child_begin+1);
                else policy[node][i]=max(regret_plus[node][i],.0)/regret_plus_sum[tree[node].fa][i];
                hand_prob[player][node][i]=hand_prob[player][tree[node].fa][i]*policy[node][i];
                hand_prob[1-player][node][i]=hand_prob[1-player][tree[node].fa][i];
            }else hand_prob[0][node][i]=hand_prob[1][node][i]=0.0;//R6
        }//R5
        int ite=0;
        for(int node=0;node<TREE_SZ;node++){//L5
            int player=tree[node].ps.player;
            if(tree[node].ps.type==FOLD_PUBLIC_STATE){//L6
                for(int pl=0;pl<2;pl++){//L7
                    for(int i=0;i<hand_cnt;i++){
                        double other_prob=0.0;
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        for(int j=0;j<hand_cnt;j++){
                            int j1=index1[handid[j]],j2=index2[handid[j]];
                            if(i1!=j1&&i1!=j2&&i2!=j1&&i2!=j2)other_prob+=hand_prob[1-pl][node][i];
                        }
                        if(pl==player)hand_value[pl][node][i]=node.ps.totalv/2*other_prob;
                        else hand_value[pl][node][i]=-node.ps.totalv/2*other_prob;
                    }
                }//R7
            }//R6
            if(tree[node].ps.type==SHOWDOWN_PUBLIC_STATE){
                for(int pl=0;pl<2;pl++){
                    for(int i=0;i<hand_cnt;i++){
                        int i1=index1[handid[i]],i2=index2[handid[i]];
                        for(int j=0;j<hand_cnt;j++){
                            int j1=index1[handid[j]],j2=index2[handid[j]];
                            if(i1!=j1&&i1!=j2&&i2!=j1&&i2!=j2){
                                hand_value[pl][node][i]+=totalv[node]/2*compare(handid[i],handid[j],public_cards);
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
        for(int node=TREE_SZ-1;node;node--)if(1-action_player[node]==up_player){
            for(int i=0;i<hand_cnt;i++){//L7
                if(regret_plus[node][i]>=0)regret_plus[node][i]*=pos_factor;else regret_plus[node][i]*=neg_factor;//DCFR
                regret_plus[node][i]+=hand_value[up_player][node][i]-hand_value[up_player][tree[node].fa][i];
                regret_plus_sum[tree[node].fa][i]+=max(regret_plus[node][i],.0);
            }//R7
        }
        if(T>=VALUE_WARM_ITERATOR){
            for(int node=0;node<TREE_SZ;node++)
                for(int i=0;i<hand_cnt;i++)//L6
                    updatev(average_value[0][node][i],hand_value[0][0][i],2.0/(T-VALUE_WARM_ITERATOR+2));
                    updatev(average_value[1][node][i],hand_value[1][root_id][i],2.0/(T-VALUE_WARM_ITERATOR+2));
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
        if(tree[root].ps.player==0)value+=average_value[0][0][i]*average_hand_prob[0][0][i];
        else value+=average_value[1][0][i]*average_hand_prob[1][0][i];
    }
    return value;
}
void add_sav_data(vector<float>state,vector<float>choose_actions,double base_value,double action_value){
    rl_state_data.push_back(state);
    if((int)choose_actions.size()!=3){
        cerr<<"ACTION NUMBER ERROR"<<(int)choose_actions.size()<<endl;
    }
    rl_action_data.push_back(choose_actions);
    rl_value_data.push_back(REVALUE((action_value-base_value)*100));
}
void train_with_action(double big_blinds,bool trust_action){
    vector<vector<double>>policy;
    vector<ACTION>actions;
    PBS ps;
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
    for(int i=0;i<HANDS_NUMBER;i++)if(FLAG[i])oop_prob[i]=1.0/cnt,ip_prob[i]=1.0/cnt;
    while(ps.type==3){
        //pbs.see();
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
            add_sav_data(state,choose_actions,base_value,action_value);
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
    string crit_dir=critic_dir+"critic.pt";
    try{
        critic_module=torch::jit::load(crit_dir,torch::Device(torch::DeviceType::CUDA,gpu_id));
        critic_module.to(device);
    }
    catch(const c10::Error& e){
        cerr<<"CRITIC MODULE LOAD ERROR"<<endl;
    }
    srand(time(0)+thread_id*23333);
}
void REBEL_training_with_action(string data_dir,int epoch,int thread_id,int sample_number){
    pre_setting(thread_id);
    for(int i=0;i<sample_number;i++){
        train_with_action(randvalue(5,200),TRUST_ACTION_FLAG);
    }
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
void REBEL_training_type(string data_dir,int type, int epoch,int thread_id,int sample_number){
	if(type==0)REBEL_training(data_dir,epoch,thread_id,sample_number);
	//if(type==1)REBEL_training_last(data_dir,epoch,thread_id,sample_number);
	//if(type==2)REBEL_training_from_flop(data_dir,epoch,thread_id,sample_number);
	if(type==3)REBEL_training_with_action(data_dir,epoch,thread_id,sample_number);
}
//turn to AI IN:state event ai_actor_player    OUT: for all hands get a policy
double AI_turn(string AI_name,PBS pbs,vector<vector<double>>event,vector<ACTION>&final_actions,vector<vector<double>>&total_policy){
    int lx=1;
    if(AI_name=="MULTI"){
        lx=1;
        double max_value=call_value(lx,DEFAULT_RAISING,0,pbs,event,final_actions,total_policy);
        vector<ACTION>max_actions=final_actions;
        vector<vector<double>>max_total_policy=total_policy;
        vector<vector<double>>raising_list={{0.25,0.5,1.0},{0.33,0.7,1.5}};
        for(vector<double>new_raising:raising_list){
            double v=call_value(lx,new_raising,0,pbs,event,final_actions,total_policy);
            if(v>max_value){
                max_value=v;
                max_actions=final_actions;
                max_total_policy=total_policy;
            }
        }
        final_actions=max_actions;
        total_policy=max_total_policy;
    }
    vector<double>raising_scale=DEFAULT_RAISING;
    double v;
    if(AI_name=="REBEL")lx=1;
    if(AI_name=="Supreme"){
        lx=1;
        raising_scale={0.25,0.5,0.75,1,1.25,2.0};
    }
    if(AI_name=="ACTIONRL1")lx=2;
    if(AI_name=="ACTIONRL2")lx=3;
    if(lx==1){
        v=call_value(lx,DEFAULT_RAISING,0,pbs,event,final_actions,total_policy);
    }
    if(lx==2||lx==3){
        choose_actions.clear();
        root_actions=get_action(pbs.stage,pbs.player,pbs.totalv,pbs.callv,pbs.maxraise,pbs.public_cards,event,0.0);
        v=call_value(lx,DEFAULT_RAISING,0,pbs,event,final_actions,total_policy);
        /*vector<ACTION>final_actions1,final_actions2;
        vector<vector<double>>total_policy1,total_policy2;
        PBS inpbs=pbs;vector<vector<double>>inevent=event;
        double v1=call_value(lx,DEFAULT_RAISING,0,inpbs,inevent,final_actions1,total_policy1);
        double v2=call_value(1,DEFAULT_RAISING,0,pbs,event,final_actions2,total_policy2);
        if(v1>v2){
            final_actions=final_actions1;
            total_policy=total_policy1;
        }else{
            final_actions=final_actions2;
            total_policy=total_policy2;
        }*/
    }
    return v;
}
ACTION AI_turn_with_hand(int hand_id,string AI_name,PBS &pbs,vector<vector<double>>&event,vector<ACTION>&final_actions,vector<vector<double>>&total_policy){
    AI_turn(AI_name,pbs,event,final_actions,total_policy);
    if((int)final_actions.size()<1)cerr<<"FINAL ACTION EMPTY ERROR"<<endl;
    if((int)total_policy.size()!=(int)final_actions.size())cerr<<"AI TURN POLICY ACTION SIZE ERROR"<<endl;
    double all_prob=.0;
    for(int i=0;i<(int)final_actions.size();i++)all_prob+=max(total_policy[i][hand_id]-BAD_ACTION_EPSON,.0);
    double epson=randvalue(0.0,all_prob),nowv=0.0;
    int action_id=0,final_optimal_action;
    ACTION selected_action=final_actions[0];
    for(int i=0;i<(int)final_actions.size();i++){
        if(SEE_ACTION_FLAG)printf("ACTION %d %.4lf   PROB %.12lf\n",final_actions[i].type,final_actions[i].raise_v,total_policy[i][hand_id]);
        if(total_policy[i][hand_id]>BAD_ACTION_EPSON){
            final_optimal_action=i;
            action_id=i;
            selected_action=final_actions[i];
        }
    }
    if((int)final_actions.size()==0)cerr<<"FINAL ERROR EMPTY"<<endl;
    for(int i=0;i<(int)final_actions.size();i++){
        nowv+=max(total_policy[i][hand_id]-BAD_ACTION_EPSON,.0);
        if(nowv>epson||i==final_optimal_action){
            action_id=i;
            selected_action=final_actions[i];
            break;
        }
    }
    if(pbs.player==0){
        double sum_prob=.0;
        for(int i=0;i<HANDS_NUMBER;i++){
            pbs.oop_prob[i]*=total_policy[action_id][i];
            sum_prob+=pbs.oop_prob[i];
        }
        if(sum_prob>1e-15){
            sum_prob=1.0/sum_prob;
            for(int i=0;i<HANDS_NUMBER;i++)pbs.oop_prob[i]*=sum_prob;
        }else{
            cerr<<"SELECT ACTION PROB ERROR"<<endl;
        }
    }
    if(pbs.player==1){
        double sum_prob=.0;
        for(int i=0;i<HANDS_NUMBER;i++){
            pbs.ip_prob[i]*=total_policy[action_id][i];
            sum_prob+=pbs.ip_prob[i];
        }
        if(sum_prob>1e-15){
            sum_prob=1.0/sum_prob;
            for(int i=0;i<HANDS_NUMBER;i++)pbs.ip_prob[i]*=sum_prob;
        }else{
            cerr<<"SELECT ACTION PROB ERROR"<<endl;
        }
    }
    if(selected_action.type==0){//FOLD
        pbs.type=0;
        pbs.totalv-=pbs.callv;
        pbs.callv=0;
        pbs.player=1-pbs.player;
    }else if(selected_action.type==1){//CHECK/CALL
        if((pbs.stage==0&&pbs.totalv>BIG_BLIND_V*1.7)||(pbs.stage>0&&pbs.player==1)||(pbs.stage>0&&pbs.callv>EPS)||(pbs.callv<EPS&&pbs.maxraise<EPS)){
            if(pbs.stage==3)pbs.type=1;else pbs.type=2,pbs.player=0;
            pbs.stage++;
        }else pbs.type=3,pbs.player=1-pbs.player;
        pbs.totalv+=pbs.callv;
        pbs.callv=0.0;
        event.back().push_back(0.0);
    }else if(selected_action.type==2){//RAISE
        event.back().push_back(1.0*selected_action.raise_v/(pbs.totalv+pbs.callv));
        pbs.totalv+=pbs.callv+selected_action.raise_v;
        pbs.callv=selected_action.raise_v;
        pbs.maxraise-=selected_action.raise_v;
        pbs.type=3;
        pbs.player=1-pbs.player;
    }else{
        cerr<<"SELECT ACTION ERROR"<<endl;
    }
    return selected_action;
}
/*
void exploitability_test(string AI_name, string EXPLO_name, int play_time,int thread_id){
    pre_setting(thread_id);
    double AI_explo_value=0.0,OPPO_explo_value=0.0;
    int AI_explo_time=0,OPPO_explo_time=0;
    for(int it=0;it<play_time;it++){//it=0 AI oop  it=1 AI ip
        int ai_hand_id=rand()%HANDS_NUMBER;
        vector<Card>player_cards={Card(index1[ai_hand_id]),Card(index2[ai_hand_id])};
        vector<vector<double>>policy;
        vector<ACTION>actions;
        PBS pbs;
        pbs.reset(200);//100BB
        vector<Card>next_cards;
        vector<vector<double>>events;
        events.clear();events.push_back({});
        while(pbs.type>=2&&pbs.stage<=3){
            if(pbs.maxraise<EPS)break;
            if(pbs.type==2){
                next_public_card(0,pbs);
                pbs.player=0;
                events.push_back({});
                continue;
            }
            if(pbs.stage==3)break;
            vector<float>state=vector_event_to_state(pbs.stage,pbs.player,pbs.totalv,pbs.callv,pbs.maxraise,pbs.public_cards,events);
            string my_AI,other_AI;
            if(it%2==pbs.player)my_AI=AI_name,other_AI=EXPLO_name;
            else my_AI=EXPLO_name,other_AI=EXPLO_name;
            if(my_AI=="ACTIONRL1"||my_AI=="ACTIONRL2"){
                choose_actions.clear();
                root_actions=get_action(pbs.stage,pbs.player,pbs.totalv,pbs.callv,pbs.maxraise,pbs.public_cards,events,0.0);
                call_value(my_AI=="ACTIONRL1"?2:3,DEFAULT_RAISING,3,pbs,events,actions,policy);
            }else{
                call_value(1,DEFAULT_RAISING,3,pbs,events,actions,policy);
            }
        }
        if(pbs.maxraise<EPS||pbs.stage<=2||pbs.type<2)continue;
        vector<float>state=vector_event_to_state(pbs.stage,pbs.player,pbs.totalv,pbs.callv,pbs.maxraise,pbs.public_cards,events);
        choose_actions.clear();
        root_actions=get_action(pbs.stage,pbs.player,pbs.totalv,pbs.callv,pbs.maxraise,pbs.public_cards,events,ACTION_NOISE);
        double explo_value=call_exploitability(pbs,events,it%2==0?0:1,it%2==pbs.player?AI_name:EXPLO_name,it%2!=pbs.player?AI_name:EXPLO_name);
        if(it%2==0)AI_explo_value+=explo_value,AI_explo_time++;else OPPO_explo_value+=explo_value,OPPO_explo_time++;
    }
    printf("%.12lf %d %.12lf %d\n",AI_explo_value,AI_explo_time,OPPO_explo_value,OPPO_explo_time);
    //double exp_value=call_value(trust_action?3:2,DEFAULT_RAISING,3,type,stage,action_player,totalv,callv,maxraise,oop_prob,ip_prob,public_cards,events,actions,policy);
}*/
pair<double,double> AI_versus_AI(string AI_name1, string AI_name2, int player_number,int thread_id){
    pre_setting(thread_id);
    double total_profit=0.0, fangcha=0.0, fangcha2=0.0, pre_profit, pre2_profit;//AI1-AI2 value
    static int ranks[HANDS_NUMBER];
    vector<Card>player_cards;
    vector<Card>same_public_cards;
    int hand_id_oop,hand_id_ip;
        for(int ite=0;ite<player_number;ite++){
            pre_profit=total_profit;
            if(ite%2==0){
                pre2_profit=total_profit;
                while(true){
                    int x=rand()%HANDS_NUMBER;
                    int y=rand()%HANDS_NUMBER;
                    if(index1[x]!=index1[y]&&index2[x]!=index1[y]&&index1[x]!=index2[y]&&index2[x]!=index2[y]){
                        hand_id_oop=x;
                        hand_id_ip=y;
                        player_cards={Card(index1[x]),Card(index2[x]),Card(index1[y]),Card(index2[y])};
                        break;
                    }
                }
                same_public_cards.clear();
                while((int)same_public_cards.size()<5){
                    bool flag=false;
                    int rdcd=rand()%CARD_NUMBER;
                    for(Card cd:player_cards)if(rdcd==cd.index)flag=true;
                    for(Card cd:same_public_cards)if(rdcd==cd.index)flag=true;
                    if(!flag)same_public_cards.push_back(Card(rdcd));
                }
            }
            puts("SAME PUBLIC CARDS:");
            for(Card cd:same_public_cards)cd.output();
            puts("");
            vector<vector<double>>policy;
            vector<ACTION>actions;
            PBS pbs;
            pbs.reset(200);//200BB
            vector<vector<double>>events;
            events.clear();
            events.push_back({});
            while(pbs.type>=2&&pbs.stage<=3){
                if(SEE_PBS_FLAG)pbs.allsee();
                //printf("%d %d %.12lf %.12lf %.12lf  %d\n",type,stage,totalv,callv,maxraise,events.size());
                //printf("PUBLIC CARDS: ");for(Card cd:public_cards)printf(" %d",cd.index);puts("");
                //printf("OOP CARDS: %d %d   IP CARDS: %d %d\n",player_cards[0].index,player_cards[1].index,player_cards[2].index,player_cards[3].index);
                if(SEE_PBS_FLAG){
                    printf("OOP CARDS:");player_cards[0].output();player_cards[1].output();puts("");
                    printf("IP CARDS :");player_cards[2].output();player_cards[3].output();puts("");
                }    
                
                if(pbs.type==2){
                    next_public_card(0,pbs,player_cards,same_public_cards);
                    pbs.player=0;
                    events.push_back({});
                    continue;
                }
                vector<float>state=vector_event_to_state(pbs.stage,pbs.player,pbs.totalv,pbs.callv,pbs.maxraise,pbs.public_cards,events);
                //ite%2==0 AI1 oop AI2 ip     ite%2==1 AI1 ip AI2 oop
                if(pbs.type==3){
                    if(pbs.player==0){
                        AI_turn_with_hand(hand_id_oop,(ite%2==0)?AI_name1:AI_name2,pbs,events,actions,policy);
                    }else{
                        AI_turn_with_hand(hand_id_ip,(ite%2==1)?AI_name1:AI_name2,pbs,events,actions,policy);
                    }
                }
            }
            if(pbs.type==0){
                if(pbs.player==0){//OOP WIN
                    if(ite%2==0)total_profit+=pbs.totalv/2;else total_profit-=pbs.totalv/2;
                }else{
                    if(ite%2==0)total_profit-=pbs.totalv/2;else total_profit+=pbs.totalv/2;
                }
            }else if(pbs.type==1){
                vector<int>vc;
                int DIFF_RANKS;
                for(Card cd:pbs.public_cards)vc.push_back(cd.index);
                sort(vc.begin(),vc.end());
                if((int)vc.size()!=5){
                    printf("%d\n",(int)vc.size());
                    cerr<<"PBCARD SIZE ERROR"<<endl;
                }
                for(int i=0;i<5;i++)if(vc[i]<0||vc[i]>=CARD_NUMBER){
                    cerr<<"PBCARD RANGE ERROR"<<endl;
                }
                string dir="../hand_rank/"+to_string(vc[0])+"_"+to_string(vc[1])+"_"+to_string(vc[2])+"_"+to_string(vc[3])+"_"+to_string(vc[4])+".txt";
                FILE *fin;
                fin=fopen(dir.c_str(),"rb");
                fscanf(fin,"%d",&DIFF_RANKS);
                for(int i=0;i<HANDS_NUMBER;i++)fscanf(fin,"%d",&ranks[i]);
                fclose(fin);
                if(ranks[hand_id_oop]<ranks[hand_id_ip]){//OOP WIN
                    if(ite%2==0)total_profit+=pbs.totalv/2;else total_profit-=pbs.totalv/2;
                }else if(ranks[hand_id_oop]>ranks[hand_id_ip]){
                    if(ite%2==0)total_profit-=pbs.totalv/2;else total_profit+=pbs.totalv/2;
                }
            }else cerr<<"TYPE ERROR"<<endl;
            //if((ite+1)%100==0)printf("%d %.12lfBB  %.12lfMBB\n",ite+1,total_profit*100,total_profit*100000/(ite+1));
            fangcha+=(total_profit-pre_profit)*(total_profit-pre_profit);
            if(ite%2==1)fangcha2+=(total_profit-pre2_profit)*(total_profit-pre2_profit);
        }
    return make_pair(total_profit,min(fangcha,fangcha2));
}
#endif
