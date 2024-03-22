#include"solver.hpp"
/*
int main(int argc, char*argv[]){//vs Slumbot
    srand(time(0));
    play_with_slumbot("REBEL",atoi(argv[1]),atoi(argv[2]));
}*/
/*
int main(){
    see_PBS();
}*/
/*
int main(int argc, char*argv[]){//training
    srand(time(0));
    REBEL_training_type("../data/"+to_string(atoi(argv[1]))+"/",atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
}*/

/*
int main(int argc, char*argv[]){//explo
    srand(time(0));
    exploitability_test("REBEL","ACTIONRL1",atoi(argv[1]),atoi(argv[2]));
}*/
/*
int main(int argc, char*argv[]){
    srand(time(0));
    ai_play_with_human("ACTIONRL2",atoi(argv[1]));
}*/
int main(int argc, char*argv[]){
    srand(time(0));
    solve({},{},0);
}
/*
int main(int argc, char*argv[]){//AI_vs_AI
    srand(time(0));
    pair<double,double>pii=AI_versus_AI("ACTIONRL1","REBEL",atoi(argv[1]),atoi(argv[2]),true);
    printf("%.12lf %.12lf\n",pii.first,pii.second);
    //double tmp=AI_versus_AI("REBEL","ACTIONRL1",atoi(argv[1]),atoi(argv[2]));
    //printf("%.12lf\n",tmp);
    //REBEL_training_type("../data/"+to_string(atoi(argv[1]))+"/",atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
}*/

/*
int main(int argc, char*argv[]){//see AI_vs_AI
    srand(time(0));
    pair<double,double>pii=AI_versus_AI("ACTIONRL1","REBEL",atoi(argv[1]),atoi(argv[2]),true);
    printf("ALLEV %.12lf %.12lf\n",pii.first,pii.second);
    //double tmp=AI_versus_AI("REBEL","ACTIONRL1",atoi(argv[1]),atoi(argv[2]));
    //printf("%.12lf\n",tmp);
    //REBEL_training_type("../data/"+to_string(atoi(argv[1]))+"/",atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
}*/

/*
int main(int argc, char*argv[]){//vs Slumbot
    srand(time(0));
    get_action("RLACTION1",atoi(argv[1]));
}*/
/*
int main(){
    get_action("REBEL");
}*/
/*#include<pthread.h>

const int NUM_THREADS=60;
int id=0;
void* train(void* args){
    id++;
    REBEL_training(id);
    return 0;
}
int main(int argc, char* argv[]){
    pthread_t tids[NUM_THREADS];
    for(int i=0;i<NUM_THREADS;i++){
        int ret=pthread_create(&tids[i],NULL,train,NULL);
        if(ret!=0)cerr<<"pthread_creat error: error_code="<<ret<<endl;
    }
    pthread_exit(NULL);
}*/
