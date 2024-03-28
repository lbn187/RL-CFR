#include"CFR.hpp"
int main(int argc, char*argv[]){//training
    srand(time(0));
    if(atoi(argv[1])==-1){
        printf("DEFAULT FIXED ACTION ABSTRACTION LOSE %.12lf mbb\n",test_abstractions(atoi(argv[2]))/atoi(argv[2])*100000);
        return 0;
    }
    if(atoi(argv[1])==-2){
        printf("Exploitability %.12lf\n",test_exploitability(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]))/atoi(argv[4]));
        return 0;
    }
    if(atoi(argv[1])==-3){
        int num=atoi(argv[5]);
        pair<double,double>uii=test_AI(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
        printf("%.12lf +- %.12lf mbb\n",uii.first/atoi(argv[5])*100000,sqrt(uii.second/num)/sqrt(1.0*num)*1.96*100000);
        return 0;
    }
    training_with_action("../data/",atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
    return 0;
}