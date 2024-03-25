#ifndef Card_H
#define Card_H
#include"utils.hpp"
int cal_suit(char ch){
    if(ch=='s')return 0;
    if(ch=='d')return 1;
    if(ch=='c')return 2;
    return 3;
}
string suit_string(int x){
    if(x==0)return "s";
    if(x==1)return "d";
    if(x==2)return "c";
    return "h";
}
int cal_number(char ch){
    if(ch=='A')return 0;
    if(ch=='K')return 1;
    if(ch=='Q')return 2;
    if(ch=='J')return 3;
    if(ch=='T')return 4;
    if(ch=='9')return 5;
    if(ch=='8')return 6;
    if(ch=='7')return 7;
    if(ch=='6')return 8;
    if(ch=='5')return 9;
    if(ch=='4')return 10;
    if(ch=='3')return 11;
    return 12;
}
string number_string(int x){
    if(x==0)return "A";
    if(x==1)return "K";
    if(x==2)return "Q";
    if(x==3)return "J";
    if(x==4)return "T";
    if(x==5)return "9";
    if(x==6)return "8";
    if(x==7)return "7";
    if(x==8)return "6";
    if(x==9)return "5";
    if(x==10)return "4";
    if(x==11)return "3";
    return "2";
}
struct Card{
    int index;
    Card(){}
    Card(int _index){index=_index;}
    Card(const Card&_card){
        index=_card.index;
    }
    Card &operator=(const Card&_cd){
        if(&_cd==this)return *this;
        index=_cd.index;
        return *this;
    }
    Card(int number,int suit){
        index=number*4+suit;
    }
    bool operator<(const Card&x){
        return index<x.index;
    }
    int suit(){
        return index%4;
    }
    int number(){
        return index/4;
    }
    string to_string(){
        return number_string(number())+suit_string(suit());
    }
    void output(){
        cout<<number_string(number())<<suit_string(suit());
    }
};
bool operator==(Card a,Card b){
        return a.index==b.index;
}
bool operator!=(Card a,Card b){
    return a.index!=b.index;
}
int get_rank(vector<Card>five_cards){
    sort(five_cards.begin(),five_cards.end());
    bool suit_flag=true;
    int stright=-1,sitiao=-1,santiao=-1,yidui=-1,liangdui=-1,A=-1,B=-1,C=-1,D=-1,E=-1;
    vector<int>cnt(14,0);
    for(int i=0;i<4;i++){
        if(five_cards[i].suit()!=five_cards[i+1].suit())suit_flag=false;
    }
    for(int i=0;i<5;i++){
        cnt[five_cards[i].number()]++;
        if(five_cards[i].number()==0)cnt[13]++;
    }
    for(int i=0;i<=9;i++)
        if(cnt[i]&&cnt[i+1]&&cnt[i+2]&&cnt[i+3]&&cnt[i+4])stright=i;
        if(stright!=-1&&suit_flag==true)return stright;
        for(int i=0;i<13;i++){
            if(cnt[i]==4)sitiao=i;
            if(cnt[i]==3)santiao=i;
            if(cnt[i]==2){
                if(yidui==-1)yidui=i;else liangdui=i;
            }
            if(cnt[i]==1){
                if(A==-1)A=i;else if(B==-1)B=i;else if(C==-1)C=i;else if(D==-1)D=i;else E=i;
            }
        }
    if(sitiao!=-1){
        int x=-1;
        for(int i=0;i<13;i++)if(cnt[i]==1)x=i;
	    int ans=sitiao*12+(x>sitiao?x-1:x);
    	if(ans<0||ans>155){cerr<<"ERROR SITIAO"<<endl;}
	    return 10+ans;
    }
    if(yidui!=-1&&santiao!=-1){
	    int ans=santiao*12+(yidui>santiao?yidui-1:yidui);
    	if(ans<0||ans>155)cerr<<"ERROR HULU"<<endl;
	    return 166+ans;
    }
    if(suit_flag==true){
        int ans=0;
        for(int i=0;i<A;i++)
            ans+=(12-i)*(11-i)*(10-i)*(9-i)/24;
        for(int i=A+1;i<B;i++)
            ans+=(12-i)*(11-i)*(10-i)/6;
        for(int i=B+1;i<C;i++)
            ans+=(12-i)*(11-i)/2;
        for(int i=C+1;i<D;i++)
            ans+=12-i;
        ans+=E-D-1;
        ans-=(A+1);
        if(A>0)ans--;
	if(ans<0||ans>1277)cerr<<"ERROR TONGHUA"<<endl;
        return 322+ans;
    }
    if(stright!=-1){
	    return 1599+stright;
    }
    if(santiao!=-1){
        if(A>santiao)A--;
        if(B>santiao)B--;
	    int ans=santiao*66+(23-A)*A/2+(B-A-1);
        if(ans<0||ans>858)cerr<<"ERROR SANTIAO"<<endl;
	    return 1609+ans;
    }
    if(liangdui!=-1){
        int ans=(yidui*(24-yidui+1)/2+(liangdui-yidui-1))*11+A-(A>yidui?1:0)-(A>liangdui?1:0);
    	if(ans<0||ans>858)cerr<<"ERROR LIANGDUI"<<endl;
	    return 2467+ans;
    }
    if(yidui!=-1){
        if(A>yidui)A--;
        if(B>yidui)B--;
        if(C>yidui)C--;
        int ans=0;
        for(int i=0;i<A;i++)
            ans+=(11-i)*(10-i)/2;
        for(int i=A+1;i<B;i++)
            ans+=11-i;
        ans+=(C-B-1);
	    ans+=yidui*220;
	    if(ans<0||ans>6185-3325)cerr<<"ERROR YIDUI"<<endl;
        return 3325+ans;
    }
    int ans=0;
    for(int i=0;i<A;i++)
        ans+=(12-i)*(11-i)*(10-i)*(9-i)/24;
    for(int i=A+1;i<B;i++)
        ans+=(12-i)*(11-i)*(10-i)/6;
    for(int i=B+1;i<C;i++)
        ans+=(12-i)*(11-i)/2;
    for(int i=C+1;i<D;i++)
        ans+=12-i;
    ans+=E-D-1;
    ans-=(A+1);
    if(A>0)ans--;
    return 6185+ans;
}
int get_rank_from_cards(vector<Card>cards){
    int N=(int)cards.size();
    int rk=7472;
    for(int a=0;a<N;a++)
        for(int b=a+1;b<N;b++)
            for(int c=b+1;c<N;c++)
                for(int d=c+1;d<N;d++)
                    for(int e=d+1;e<N;e++)
                        rk=min(rk,get_rank(vector<Card>{cards[a],cards[b],cards[c],cards[d],cards[e]}));
    return rk;
}
void get_hand_id(){
    int num=0;
    for(int id1=1;id1<CARD_NUMBER;id1++){
        for(int id2=0;id2<id1;id2++){
            index1[num]=id1;
            index2[num]=id2;
            to_index[id1][id2]=to_index[id2][id1]=num;
            Card cd1=Card(id1),cd2=Card(id2);
            start_id[num]=(cd2.suit()!=cd1.suit()?cd2.number()*13+cd1.number():cd1.number()*13+cd2.number());
            ++num;
        }
    }
}
vector<Card> dealt_cards(){
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
    return public_cards;
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
void set_prob(vector<Card>public_cards,vector<double>&oop_prob,vector<double>&ip_prob,vector<int>&hand_id){
    static bool FLAG[HANDS_NUMBER],VV[CARD_NUMBER];
    int cnt=0;
    for(int i=0;i<CARD_NUMBER;i++)VV[i]=false;
    for(Card cd:public_cards)VV[cd.index]=true;
    for(int i=0;i<HANDS_NUMBER;i++)if(VV[index1[i]]||VV[index2[i]])FLAG[i]=false;else FLAG[i]=true,cnt++;
    oop_prob.resize(HANDS_NUMBER,0.0);
    ip_prob.resize(HANDS_NUMBER,0.0);
    hand_id.clear();
    for(int i=0;i<HANDS_NUMBER;i++)if(FLAG[i]){
        oop_prob[i]=1.0/cnt,ip_prob[i]=1.0/cnt;
        hand_id.push_back(i);
    }
    for(int i=0;i<(int)hand_id.size();i++)
        for(int j=0;j<(int)hand_id.size();j++)
            compare_ans[i][j]=compare(hand_id[i],hand_id[j],public_cards);
}
#endif
