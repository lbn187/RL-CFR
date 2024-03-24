#include"CFR.hpp"

int main(int argc, char*argv[]){//training
    srand(time(0));
    if(atoi(argv[1])==-1){
        printf("%.12lf\n",test_abstractions(atoi(argv[2])));
        return 0;
    }
    if(atoi(argv[1])==-2){
        printf("%.12lf\n",test_exploitability(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]))/atoi(argv[4]));
        return 0;
    }
    training_with_action("../data/",atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
    return 0;
}

/*
int main(int argc, char*argv[]){//explo
    srand(time(0));
    exploitability_test("REBEL","ACTIONRL1",atoi(argv[1]),atoi(argv[2]));
}*/

/*
int main(int argc, char*argv[]){//AI_vs_AI
    srand(time(0));
    pair<double,double>pii=AI_versus_AI("ACTIONRL1","REBEL",atoi(argv[1]),atoi(argv[2]),true);
    printf("%.12lf %.12lf\n",pii.first,pii.second);
    //double tmp=AI_versus_AI("REBEL","ACTIONRL1",atoi(argv[1]),atoi(argv[2]));
    //printf("%.12lf\n",tmp);
    //REBEL_training_type("../data/"+to_string(atoi(argv[1]))+"/",atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
}*/
