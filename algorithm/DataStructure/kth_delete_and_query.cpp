#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct Node {
    int l, r;
    int minv, maxv;
};

class SegTree {
private:
    int N;
    int len;
    vector<int> arr;   // record mapped index
    vector<int> sorted;   // record original sorted array
    vector<int> sub;  // lazy mark
    vector<Node> tree;    // segment tree

    void push_up (int root) {
        tree[root].maxv = max(tree[root<<1].maxv, tree[root<<1|1].maxv);
        tree[root].minv = min(tree[root<<1].minv, tree[root<<1|1].minv);
    }

    void build(int root, int left, int right) {
        tree[root].l = left;
        tree[root].r = right;
        if (left == right) {
            tree[root].minv = tree[root].maxv = arr[left];
            return;
        }

        int mid = (left + right) >> 1;
        build(root<<1, left, mid);
        build(root<<1|1, mid + 1, right);
        push_up(root);
    }

    void push_down(int root) {
        if (sub[root]) {
            // push down first 
            sub[root<<1] += sub[root];
            sub[root<<1|1] += sub[root];

            // update child 
            tree[root<<1].maxv -= sub[root];
            tree[root<<1|1].maxv -= sub[root];
            tree[root<<1].minv -= sub[root];
            tree[root<<1|1].minv -= sub[root];

            // clear next
            sub[root] = 0;
        }
    }

    void range_update(int root, int L, int R) {
        if (L <= tree[root].l && tree[root].r <= R) {
            tree[root].maxv -= 1;
            tree[root].minv -= 1;
            sub[root] += 1;
            return;
        }

        // push down every update
        push_down(root);
        int mid = (tree[root].l + tree[root].r) >> 1;
        if (L <= mid) {
            range_update(root<<1, L, R);
        }
        if (mid < R) {
            range_update(root<<1|1, L, R);
        }
        push_up(root);
    }

    int query(int root, int k) {
        if (tree[root].maxv <= k) {
            return tree[root].r - tree[root].l + 1;
        }

        push_down(root);
        int mid = (tree[root].l + tree[root].r) >> 1;
        int res = 0;
        res += query(root<<1, k);
        if (k >= tree[root<<1|1].minv) {
            res += query(root<<1|1, k);
        }

        return res;
    }

public:
    SegTree(vector<int>& nums) {
        // init array
        N = nums.size();
        len = N;
        arr.resize(N);
        sorted.resize(N);
        tree.resize(N<<2);
        sub.resize(N<<2, 0);
        copy(nums.begin(), nums.end(), sorted.begin());

        // make table
        sort(sorted.begin(), sorted.end());
        unordered_map<int, int> table;
        for (int i = 0; i < sorted.size(); ++ i) {
            table[sorted[i]] = i;
            arr[i] = i;
        }

        build(1, 0, N - 1);
    }

    void delete_kth(int k) {
        int cnt_k = query(1, k);
        // cout << "real position of k: " << cnt_k << endl;
        range_update(1, cnt_k, N - 1);
        -- len;
    }

    int query(int k) {
        if (k >= len) {
            cout << "query idx out of range" << endl;
            return -1;
        }

        int cnt = query(1, k);
        // cout << "total cnt: " << cnt << endl;
        return sorted[cnt - 1];
    }

    void print_tree() {
        for (auto &n : tree) {
            cout << "(" << n.minv << ","<< n.maxv << ")" << ",";
        }
        cout << endl;
    }
};

int main() {
    vector<int> nums = {5,3,1,9,8,6,11,14,15};
    SegTree seg(nums);

    cout << "original array: ";
    for (auto &n : nums) cout << n << ",";
    cout << endl;

    sort(nums.begin(), nums.end());
    cout << "sorted array: ";
    for (auto &n : nums) cout << n << ",";
    cout << endl;

    while (true) {
        cout << "Press 1 : query | Press 2 : delete | Press other quit" << endl;
        int btn;
        cin >> btn;

        bool flag = false;
        int k;
        switch(btn) {
            case 1:
                cout << "query kth biggest number in array:" << endl;
                cin >> k;
                cout << "kth number is: " << seg.query(k - 1) << endl;
                seg.print_tree();
                break;
            case 2:
                cout << "delete kth biggest number in array:" << endl;
                cin >> k;
                seg.delete_kth(k - 1);
                seg.print_tree();
                break;
            default:
                flag = true;
                break;
        }
        if (flag) {
            break;
        }
    }

    return 0;
}
