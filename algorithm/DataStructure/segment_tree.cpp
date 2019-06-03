#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <climits>

using namespace std;

struct Node {
    int l,r;
    long long sum;
    int mid() {
        return (l + r) >> 1;
    }
};

class Segment_tree {
    vector<int> arr;
    vector<Node> tree;
public:
    Segment_tree(vector<int>& nums) {
        arr.resize(nums.size() + 1);
        tree.resize(4 * nums.size());
        copy(nums.begin(), nums.end(), arr.begin() + 1);

        build(1, 1, nums.size());
    }

    void push_up(int root) {
        tree[root].sum = tree[root<<1].sum + tree[root<<1|1].sum;
    }

    void build(int root, int left, int right) {
        tree[root].l = left;
        tree[root].r = right;
        if (left == right) {
            tree[root].sum = arr[left];
            return;
        }

        int mid = tree[root].mid();
        build(root<<1, left, mid);
        build(root<<1|1, mid + 1, right);
        push_up(root);
    }

    // add C
    void update(int root, int pos, long long C) {
        if (tree[root].l == tree[root].r) {
            tree[root].sum += C;
            return;
        }

        int mid = tree[root].mid();
        if (pos <= mid) {
            update(root<<1, pos, C);
        }else {
            update(root<<1|1, pos, C);
        }
        push_up(root);
    }

    long long query(int root, int L, int R) {
        if (L <= tree[root].l && tree[root].r <= R) {
            return tree[root].sum;
        }

        int mid = tree[root].mid();
        long long res = 0;
        if (L <= mid) {
            res += query(root<<1, L, R);
        }
        if (mid < R) {
            res += query(root<<1|1, L, R);
        }

        return res;
    }
};

int main() {
    vector<int> nums = {1,2,3,4,5,6,7,8,9,10};
    Segment_tree seg(nums);

    cout << "original array: " << endl;
    for (auto &n : nums) cout << n << ",";
    cout << endl;

    while (true) {
        cout << "please input query" << endl;
        char ch;
        cin >> ch;
        if (ch == 'q') {
            int l, r;
            cin >> l >> r;
            cout << "range sum is: " << seg.query(1, l, r) << endl;
        }else if (ch == 'u') {
            int pos;
            long long val;
            cin >> pos >> val;
            seg.update(1, pos, val);
            cout << "finish update" << endl;
        }else {
            break;
        }
    }

    return 0;
}
