#include<cstdio>
#include<iostream>
#include<string>
#include<cstring>
#include<algorithm>
#include<cstdlib>
#include<cmath>
#include<stack>
#include<vector>
#include<queue>
#include<map>
#include<ctime>
using namespace std;
long long a[100005];
int main(){
    int i,j,m,n;
    long long now,ans,vmax;
    while(scanf("%d%d",&n,&m)!=EOF){
        for(i=1;i<=n;i++){
            scanf("%lld",&a[i]);
        }
        vmax=0;
        now = 0;
        ans = 0;
        for(i=1;i<=n;i++){
            if(now>=m){
                break;
            }
            now+=a[i];
            vmax=max(vmax,now);
            ans++;
        }
        if(now>=m){
            printf("%lld\n",ans);
        }
        else{
            if(now<=0) printf("-1\n");
            else{
                ans = ceil(1.0*(m-vmax)/now) * n;
                now = ceil(1.0*(m-vmax)/now) * now;
                for(i=1;i<=n;i++){
                    if(now>=m){
                        break;
                    }
                    now+=a[i];
                    vmax=max(vmax,now);
                    ans++;
                }
                printf("%lld\n",ans);
            }
        }
    }
    system("pause");
    return 0;
} 
