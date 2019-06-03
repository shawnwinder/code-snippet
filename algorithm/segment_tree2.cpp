#include <iostream>
#include <vector>
#include <algorithm>

#define maxn 10005

using namespace std;

struct Node {
    int l, r;
    int sum;
    int add;
    int mid() {
        return (l + r) >> 1;
    }
};

class SegmentTree {
    vector<int> nums;
    vector<Node> tree;
    int N;
public:
    SegmentTree(vector<int>& arr) {
        N = arr.size();
        nums.resize(N);
        copy(arr.begin(), arr.end(), nums.begin());
        tree.resize(4*N);
        build_tree(1, 0, N - 1);
    }

    void build_tree(int root, int left, int right) {
        tree[root].l = left;
        tree[root].r = right;
        tree[root].add = 0;
        if (left == right) {
            tree[root].sum = nums[left];
            return;
        }

        int mid = tree[root].mid();
        build_tree(root<<1, left, mid);
        build_tree(root<<1|1, mid + 1, right);
        push_up(root);
    }
    
    void push_up(int root) {
        tree[root].sum = tree[root<<1].sum + tree[root<<1|1].sum;
    }
    
    // add val, not replace with val
    void update(int root, int pos, int val) {
        if (tree[root].l == tree[root].r) {
            tree[root].sum += val;
            return;
        }

        int mid = tree[root].mid();
        if (pos <= mid) {
            update(root<<1, pos, val);
        }else {
            update(root<<1|1, pos, val);
        }
        push_up(root);
    }
    
    // not used here
    void range_update(int root, int L, int R, int val) {
        if (L <= tree[root].l && tree[root].r <= R) {
            tree[root].sum += (tree[root].r - tree[root].l + 1) * val;
            tree[root].add += val;
            return;
        }

        push_down(root);
        int mid = tree[root].mid();
        if (L <= mid) {
            range_update(root<<1, L, R, val);
        }
        if (mid < R) {
            range_update(root<<1|1, L, R, val);
        }
        push_up(root);
    }

    void push_down(int root) {
        if (tree[root].add) {
            tree[root<<1].add += tree[root].add;
            tree[root<<1|1].add += tree[root].add;

            tree[root<<1].sum = (tree[root<<1].r - tree[root<<1].l + 1) * tree[root].add;
            tree[root<<1|1].sum = (tree[root<<1|1].r - tree[root<<1|1].l + 1) * tree[root].add;

            tree[root].add = 0;
        }
    }
    
    int query(int root, int L, int R) {
        if (L <= tree[root].l && tree[root].r <= R) {
            return tree[root].sum;
        }

        int mid = tree[root].mid();
        push_down(root);
        int ans = 0;
        if (L <= mid) {
            ans += query(root<<1, L, R);
        }
        if (mid < R) {
            ans += query(root<<1|1, L, R);
        }
        return ans;
    }

    void print_tree() {
        for (int i = 0; i < 4*N; ++ i) {
            cout << tree[i].sum << ",";
        }
        cout << endl;
    }
};

int main() {
    // init
    vector<int> arr = {1,2,3,4,5,6,7,8,9,10};
    SegmentTree seg(arr);

    // apply
    cout << "original array: " << endl;
    for (auto n : arr) cout << n << ",";
    cout << endl;
    seg.print_tree();

    while (true) {
        cout << "Input (q l r) for range sum query, (u pos val) for point update,"
           " (r l r val) for range update, otherwise quit" << endl;
        char opt;
        cin >> opt;
        // range sum query
        if (opt == 'q') {
            cout << "please input query range: (l,r)" << endl;
            int l, r;
            cin >> l >> r;
            cout << "range sum is: " << seg.query(1, l, r) << endl;;
        }
        // position update
        else if (opt == 'u') {
            cout << "pleas input update position and value: (pos, val)" << endl;
            int pos, val;
            cin >> pos >> val;
            seg.update(1, pos, val);
            cout << "after update, the array is: " << endl;
            seg.print_tree();
        }else if (opt == 'r') {
            cout << "pleas input update range and value: (pos, val)" << endl;
            int l, r, val;
            cin >> l >> r >> val;
            seg.range_update(1, l, r, val);
            cout << "after update, the array is: " << endl;
            seg.print_tree();

        }else {
            break;
        }
    }

    return 0;
}
