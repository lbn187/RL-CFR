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
    if(atoi(argv[1])==-3){
        pair<double,double>uii=test_AI(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
        printf("%.12lf %.12lf\n",uii.first,uii.second);
        return 0;
    }
    training_with_action("../data/",atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));
    return 0;
}