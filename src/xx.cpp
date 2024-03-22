#include<bits/stdc++.h>
using namespace std;
int main(){
    for(int i=0;i<100;i++){
        string dir="../data/ai_rebel/pbs"+to_string(i)+".txt";
        freopen(dir.c_str(),"w",stdout);
        puts("-1");
        fclose(stdout);
    }
}
