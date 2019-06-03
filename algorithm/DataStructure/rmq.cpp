#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#define maxn 100005

using namespace std;

int NN;
int nums[maxn];
struct {
    int l, r;
    int add;
    int minv;
}tree[maxn<<2];

void push_up(int root) {
    tree[root].minv = min(tree[root<<1].minv, tree[root<<1|1].minv);
}

void build(int root, int left, int right) {
    tree[root].l = left;
    tree[root].r = right;
    tree[root].add = 0;
    if (left == right) {
        tree[root].minv = nums[left];
        return;
    }
    
    int mid = (left + right) >> 1;
    build(root<<1, left, mid);
    build(root<<1|1, mid + 1, right);
    push_up(root);
}

void push_down(int root) {
    if (tree[root].add) {
        tree[root<<1].add += tree[root].add
        tree[root<<1|1].add += tree[root].add
        
        tree[root<<1].minv += tree[root].add;
        tree[root<<1|1].minv += tree[root].add;
        tree[root].add = 0;
    }
}

void range_update(int root, int L, int R, int C) {
    if (L <= tree[root].l && tree[root].r <= R) {
        tree[root].minv += C;
        tree[root].add += C;
        return;
    }
    
    push_down(root);
    int mid = (tree[root].l + tree[root].r) >> 1;
    if (L <= mid) {
        range_update(root<<1, L, R, C);
    }
    if (mid < R) {
        range_update(root<<1|1, L, R, C);
    }
    push_up(root);
}

int query(int root, int L, int R) {
    if (L <= tree[root].l && tree[root].r <= R) {
        return tree[root].minv;
    }
    
    push_down(root);
    
    int res = INT_MAX;
    int mid = (tree[root].l + tree[root].r) >> 1;
    if (L <= mid) {
        res = min(res, query(root<<1, L, R));
    }
    if (mid < R) {
        res = min(res, query(root<<1|1, L, R));
    }
    return res;
}

void print_tree() {
    for (int i = 0; i < 4*NN; ++ i) {
        cout << tree[i].minv << ",";
    }
    cout << endl;
}

int main() {
    int N, Q;
    cin >> N >> Q;
    NN = N;
    for (int i = 0; i < N; ++ i) {
        cin >> nums[i];
    }
    build(1, 0, N - 1);
    
    // handle queries
    for (int i = 0; i < Q; ++ i) {
        char ch;
        cin >> ch;
        if (ch == 'q') {
            int l, r;
            cin >> l >> r;
            cout << query(1, l - 1, r - 1) << endl;
        }else {
            int x, y, z;
            cin >> x >> y >> z;
            range_update(1, x - 1, y - 1, z);
        }
        print_tree();
    }
    
    return 0;
}
