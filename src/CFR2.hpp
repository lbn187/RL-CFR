
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