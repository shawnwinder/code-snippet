#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <cstring>

using namespace std;

struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

TreeNode* build_tree(vector<int>& v, int beg, int end) {
    if (end < beg) {
        return nullptr;
    }

    int mid = (beg + end) >> 1;
    TreeNode* root = new TreeNode(v[mid]);
    root->left = build_tree(v, beg, mid - 1);
    root->right = build_tree(v, mid + 1, end);

    return root;
}

void print_tree_by_level(TreeNode* root) {
    if (!root) {
        cout << "empty tree" << endl;
    }

    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        auto cur = q.front();
        q.pop();

        cout << cur->val << ",";
        if (cur->left) {
            q.push(cur->left);
        }
        if (cur->right) {
            q.push(cur->right);
        }
    }

    cout << endl;
}

class Solution {
public:
    // level traverse
    char* Serialize(TreeNode *root) {    
        string str;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            auto first = q.front();
            q.pop();
            if (first) {
                str += to_string(first->val);
                str.push_back('#');
                q.push(first->left);
                q.push(first->right);
            }else {
                str += "N#";
            }
        }

        char* res = new char[str.size() + 1];
        strcpy(res, str.c_str());
        return res;
    }

    TreeNode* Deserialize(char *str) {
        if (str[0] == 'N') {
            return nullptr;
        }
        
        string s = str;
        int beg = 0, end = 0;
        while (end < s.size()) {
            if (s[end] != '#') {
                ++ end;
            }else {
                break;
            }
        }
        
        TreeNode* root = new TreeNode(stoi(s.substr(beg, end)));
        beg = end + 1;
        end = beg;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty() && end < s.size()) {
            auto first = q.front();
            q.pop();

            // left
            if (s[beg] == 'N') {
                first->left = nullptr;
                beg += 2;
                end = beg;
            }else {
                while (end < s.size()) {
                    if (s[end] != '#') {
                        ++ end;
                    }else {
                        break;
                    }
                }
                string num_s = s.substr(beg, end - beg);
                beg = end + 1;
                end = beg;
                first->left = new TreeNode(stoi(num_s));
                q.push(first->left);
            }
                    
            // right
            if (s[beg] == 'N') {
                first->right = nullptr;
                beg += 2;
                end = beg;
            }else {
                while (end < s.size()) {
                    if (s[end] != '#') {
                        ++ end;
                    }else {
                        break;
                    }
                }
                string num_s = s.substr(beg, end - beg);
                beg = end + 1;
                end = beg;
                first->right = new TreeNode(stoi(num_s));
                q.push(first->right);
            }
        }

        return root;
    }
};


int main() {
    vector<int> v = {1,2,3,4,5,6,7,8};
    TreeNode* root = build_tree(v, 0, v.size() - 1);

    Solution sol;
    char* str = sol.Serialize(root);
    cout << "str: " << str << endl;
    
    TreeNode* rt = sol.Deserialize(str);
    print_tree_by_level(rt);

    return 0;
}
