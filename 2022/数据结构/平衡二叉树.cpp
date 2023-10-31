#include <iostream>
#include <Cstdlib>
#include <iostream>
#include <cmath> 
using namespace std;
template< class E>
struct Node {
    E data;
    struct Node* LCh;
    struct Node* RCh;
    int balanceFctor;           //平衡因子
};
template< class E>
class BalanceBiTree {
public:
    BalanceBiTree(Node<E>*& T);                         //初始化
    static void menu();                                 //菜单
    void destory(Node<E>*& T);                        //销毁二叉树
    void insert(Node<E>*& T, Node<E>* S);   		//将指针S所指节点插入二叉排序中
    int BiTreeDepth(Node <E>* T);                  //求树的高度
    int getNodeFactor(Node<E>* T);                 //求树中节点的平衡因子
    void factorForTree(Node<E>*& T);
    //求树中的每个节点的平衡因子
    void nodeFctorIsTwo(Node<E>*& T, Node<E>*& p);
    //获得平衡因子为2或-2的节点
    void nodeFctorIsTwoFather(Node<E>*& T, Node<E>*& f);
    //获得平衡因子为2或-2的节点的父节点
    void LLAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f);
    void LRAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f);
    void RLAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f);
    void RRAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f);
    void AllAdjust(Node<E>*& T);
    //更新平衡因子
    void BiTreeToArray(Node <E>* T, E A[], int i, int& count);
    //二叉树转数组
    void LevelTraverse(Node <E>* T, E B[], int num);
    //对二叉链表表示的二叉树，按从上到下，从左到右打印结点值，即按层次打印
    void createSubBalanceBiTree(Node<E>*& T);          //交互创建二叉平衡树
    void search(Node <E>*& T, Node <E>*& p, E x);
    //查找元素x
    Node <E>* getElementFatherPointer(Node <E>*& T, Node <E>*& f, E x);
    //获取某个元素的父亲指针，不存在返回NULL
    void getPriorElement(Node <E>*& T, E& min, E& max);                 	//获取前驱元素
    Node <E>* getElementPriorPointer(Node <E>*& T);
    //获取某个元素的前驱指针
    void getNextElement(Node <E>*& T, E& min, E& max);                  	//获取后继元素
    Node <E>* getElementNextPointer(Node <E>*& T);
    //获取某个元素的后继指针
    void deleteLeafNode(Node <E>*& T, Node <E>*& p, Node <E>*& f);
    //删除叶子节点
    void deleteO(Node <E>*& T, Node <E>*& p, Node <E>*& f);
    //删除仅有左子树或只有右子树的节点
    void deleteT(Node <E>*& T, Node <E>*& p);
    //删除既有左子树又有右子树的节点
    void alldelete(Node <E>*& T, E x);
private:
    Node<E>* root;   //树根
};
template< class E>
BalanceBiTree<E>::BalanceBiTree(Node<E>*& T)
{
    T = NULL;
}
template< class E>
void BalanceBiTree<E>::menu()
{
    cout << "1构建二叉平衡树" << endl;
    cout << "2查找" << endl;
    cout << "3删除" << endl;
    cout << "4输出" << endl;
    cout << "0销毁平衡二叉树" << endl;
}
template< class E>
void BalanceBiTree<E>::destory(Node<E>*& T)
{
    if (T)
    {
        destory(T->LCh);
        destory(T->RCh);
        delete T;
    }
}
template< class E>
void BalanceBiTree<E>::insert(Node<E>*& T, Node<E>* S)
{
    if (T == NULL)
        T = S;
    else if (S->data < T->data)
        insert(T->LCh, S);
    else
        insert(T->RCh, S);
}
template< class E>
int BalanceBiTree<E>::BiTreeDepth(Node <E>* T)
{
    int m, n;
    if (T == NULL)
        return 0;           //空树，高度为0
    else {
        m = BiTreeDepth(T->LCh);   //求左子树高度（递归）
        n = BiTreeDepth(T->RCh);   //求右子树高度（递归）
        if (m > n)
        {
            return m + 1;
        }
        else {
            return n + 1;
        }
    }
}
template< class E>
int BalanceBiTree<E>::getNodeFactor(Node<E>* T)
{
    int m = 0, n = 0;
    if (T)
    {
        m = BiTreeDepth(T->LCh);
        n = BiTreeDepth(T->RCh);
    }
    return m - n;
}
template< class E>
void BalanceBiTree<E>::factorForTree(Node<E>*& T)
{
    if (T)
    {
        T->balanceFctor = getNodeFactor(T);
        factorForTree(T->LCh);
        factorForTree(T->RCh);
    }
}
template< class E>
void BalanceBiTree<E>::nodeFctorIsTwo(Node<E>*& T, Node<E>*& p)
{
    if (T)
    {
        if (T->balanceFctor == 2 || T->balanceFctor == -2)
        {
            p = T;
        }
        nodeFctorIsTwo(T->LCh, p);
        nodeFctorIsTwo(T->RCh, p);
    }
}
template< class E>
void BalanceBiTree<E>::nodeFctorIsTwoFather(Node<E>*& T, Node<E>*& f)
{
    if (T)
    {
        if (T->LCh != NULL)
        {
            if (T->LCh->balanceFctor == 2 || T->LCh->balanceFctor == -2)
            {
                f = T;
            }
        }
        if (T->RCh != NULL)
        {
            if (T->RCh->balanceFctor == 2 || T->RCh->balanceFctor == -2)
            {
                f = T;
            }
        }
        nodeFctorIsTwoFather(T->LCh, f);
        nodeFctorIsTwoFather(T->RCh, f);
    }
}
template< class E>
void BalanceBiTree<E>::LLAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f)
{
    Node<E>* r;
    if (T == p)           //->balanceFctor==2&&T->LCh->balanceFctor!=2
    {
        T = p->LCh;        //将P的左孩子提升为新的根节点
        r = T->RCh;
        T->RCh = p;        //将p降为其左孩子的右孩子
        p->LCh = r;        //将p原来的左孩子的右孩子连接其p的左孩子

    }
    else {
        if (f->LCh == p)     //f的左孩子是p
        {
            f->LCh = p->LCh;        //将P的左孩子提升为新的根节点
            r = f->LCh->RCh;
            f->LCh->RCh = p;        //将p降为其左孩子的右孩子
            p->LCh = r;        //将p原来的左孩子的右孩子连接其p的左孩子
        }
        if (f->RCh == p)     //f的左孩子是p
        {
            f->RCh = p->LCh;        //将P的左孩子提升为新的根节点
            r = f->RCh->RCh;
            f->RCh->RCh = p;        //将p降为其左孩子的右孩子
            p->LCh = r;        //将p原来的左孩子的右孩子连接其p的左孩子
        }
    }
}
template< class E>
void BalanceBiTree<E>::LRAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f)
{
    Node<E>* l, * r;
    if (T == p)           //->balanceFctor==2&&T->LCh->balanceFctor!=2
    {
        T = p->LCh->RCh;    //将P的左孩子的右孩子提升为新的根节点
        r = T->RCh;
        l = T->LCh;
        T->RCh = p;
        T->LCh = p->LCh;
        T->LCh->RCh = l;
        T->RCh->LCh = r;
    }
    else {
        if (f->RCh == p)     //f的左孩子是p
        {
            f->RCh = p->LCh->RCh;    //将P的左孩子的右孩子提升为新的根节点
            r = f->RCh->RCh;
            l = f->RCh->LCh;
            f->RCh->RCh = p;
            f->RCh->LCh = p->LCh;
            f->RCh->LCh->RCh = l;
            f->RCh->RCh->LCh = r;
        }
        if (f->LCh == p)     //f的左孩子是p
        {
            f->LCh = p->LCh->RCh;    //将P的左孩子的右孩子提升为新的根节点
            r = f->LCh->RCh;
            l = f->LCh->LCh;
            f->LCh->RCh = p;
            f->LCh->LCh = p->LCh;
            f->LCh->LCh->RCh = l;
            f->LCh->RCh->LCh = r;
        }
    }
}
template< class E>
void BalanceBiTree<E>::RLAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f)
{
    Node<E>* l, * r;
    if (T == p)           //->balanceFctor==-2&&T->RCh->balanceFctor!=-2
    {
        T = p->RCh->LCh;
        r = T->RCh;
        l = T->LCh;
        T->LCh = p;
        T->RCh = p->RCh;
        T->LCh->RCh = l;
        T->RCh->LCh = r;
    }
    else {
        if (f->RCh == p)     //f的左孩子是p
        {
            f->RCh = p->RCh->LCh;
            r = f->RCh->RCh;
            l = f->RCh->LCh;
            f->RCh->LCh = p;
            f->RCh->RCh = p->RCh;
            f->RCh->LCh->RCh = l;
            f->RCh->RCh->LCh = r;
        }
        if (f->LCh == p)     //f的左孩子是p
        {
            f->LCh = p->RCh->LCh;
            r = f->LCh->RCh;
            l = f->LCh->LCh;
            f->LCh->LCh = p;
            f->LCh->RCh = p->RCh;
            f->LCh->LCh->RCh = l;
            f->LCh->RCh->LCh = r;
        }
    }
}
template< class E>
void BalanceBiTree<E>::RRAdjust(Node<E>*& T, Node<E>*& p, Node<E>*& f)
{
    Node<E>* l;
    if (T == p)                   //->balanceFctor==-2&&T->RCh->balanceFctor!=-2
    {
        T = p->RCh;        //将P的右孩子提升为新的根节点
        l = T->LCh;
        T->LCh = p;        //将p降为其右孩子的左孩子
        p->RCh = l;        //将p原来的右孩子的左孩子连接其p的右孩子
    //注意：p->RCh->balanceFctor==0插入节点时用不上，删除节点时可用
    }
    else {
        if (f->RCh == p)     //f的右孩子是p
        {
            f->RCh = p->RCh;        //将P的右孩子提升为新的根节点
            l = f->RCh->LCh;
            f->RCh->LCh = p;        //将p降为其右孩子的左孩子
            p->RCh = l;        //将p原来的右孩子的左孩子连接其p的右孩子
        }
        if (f->LCh == p)     //f的左孩子是p
        {
            f->LCh = p->RCh;        //将P的左孩子提升为新的根节点
            l = f->LCh->LCh;
            f->LCh->LCh = p;        //将p降为其左孩子的左孩子
            p->RCh = l;        //将p原来的右孩子的左孩子连接其p的右孩子
        }
    }
}
template< class E>
void BalanceBiTree<E>::AllAdjust(Node<E>*& T)
{
    Node<E>* f = NULL, * p = NULL;
    factorForTree(T);
    nodeFctorIsTwoFather(T, f);
    nodeFctorIsTwo(T, p);
    while (p)
    {
        factorForTree(T);
        if (p->balanceFctor == 2 && (p->LCh->balanceFctor == 1 || p->LCh->balanceFctor == 0))
        {
            LLAdjust(T, p, f);
            factorForTree(T);
        }
        else if (p->balanceFctor == 2 && p->LCh->balanceFctor == -1)
        {
            LRAdjust(T, p, f);
            factorForTree(T);
        }
        else if (p->balanceFctor == -2 && p->RCh->balanceFctor == 1)
        {
            RLAdjust(T, p, f);
            factorForTree(T);
        }
        else if (p->balanceFctor == -2 && (p->RCh->balanceFctor == -1 || p->RCh->balanceFctor == 0))  //||p->RCh->balanceFctor==0
        {
            RRAdjust(T, p, f);
        }
        f = NULL;
        p = NULL;
        nodeFctorIsTwoFather(T, f);
        nodeFctorIsTwo(T, p);
    }
}
template<class E>
void BalanceBiTree<E>::BiTreeToArray(Node <E>* T, E A[], int i, int& count)
{
    if (T != NULL)
    {
        A[i] = T->data;
        if (i > count)
            count = i;
        BiTreeToArray(T->LCh, A, 2 * i, count);
        BiTreeToArray(T->RCh, A, 2 * i + 1, count);
    }
}
template<class E>
void BalanceBiTree<E>::LevelTraverse(Node <E>* T, E B[], int num)
{
    int n, i, j, t, q, s, p, m = 0, k = 0;
    n = (int)((log(num) / log(2)) + 1);
    p = n;
    for (i = 0; i < n; i++)
    {
        k += pow(2, m) ;
        t = pow(2, m);
        j = pow(2, p - 1) - 1;
        q = pow(2, p) - 1;
        s = q;
        for (j; j > 0; j--)
        {
            cout << " ";
        }
        for (t; t <= k; t++)
        {
            if (B[t] == 0)
            {
                cout << "*";
                for (q; q > 0; q--)
                    cout << " ";
                q = s;
            }
            else {
                cout << B[t];
                for (q; q > 0; q--)
                    cout << " ";
                q = s;
            }
        }
        m++;
        p--;
        j = n - i - 1;
        cout << endl;
    }
}
template< class E>
void BalanceBiTree<E>::createSubBalanceBiTree(Node<E>*& T)
{
    int level = 1;
    int i = 1, j = 0;
    int A[100] = { 0 };
    int length = 0;
    E x;
    Node<E>* S, * p;
    T = new Node<E>;
    T->balanceFctor = 0;
    T->LCh = NULL;
    T->RCh = NULL;
    p = T;
    cout << "请输入元素(-9999退出)：";
    cin >> x;
    T->data = x;
    while (x != -9999)
    {
        cout << "请输入元素：";
        cin >> x;
        if (x == -9999)
            return;
        S = new Node<E>;
        S->data = x;
        S->balanceFctor = 0;
        S->LCh = NULL;
        S->RCh = NULL;
        insert(p, S);
        AllAdjust(T);
        p = T;
        cout << endl;
        BiTreeToArray(T, A, i, length);
        cout << "其树状图为：" << endl;
        LevelTraverse(T, A, length);
        j = 0;
        for (j; j < 100; j++)
            A[j] = 0;
        level = 1;
        i = 1;
    }
}
template<class E>
void BalanceBiTree<E>::search(Node <E>*& T, Node <E>*& p, E x)
{
    if (T)
    {
        if (T->data == x)
            p = T;
        search(T->LCh, p, x);
        search(T->RCh, p, x);
    }
}
template<class E>
Node <E>* BalanceBiTree<E>::getElementFatherPointer(Node <E>*& T, Node <E>*& f, E x)
{
    if (T)
    {
        if (T->LCh != NULL)
        {
            if (T->LCh->data == x)
                f = T;
        }
        if (T->RCh != NULL)
        {
            if (T->RCh->data == x)
                f = T;
        }
        getElementFatherPointer(T->LCh, f, x);
        getElementFatherPointer(T->RCh, f, x);
    }
    return f;
}
template<class E>
void BalanceBiTree<E>::getPriorElement(Node <E>*& T, E& min, E& max)
{
    if (T)
    {
        min = T->data;
        if (min > max)
            max = min;
        getPriorElement(T->LCh, min, max);
        getPriorElement(T->RCh, min, max);
    }
}
template<class E>
Node <E>* BalanceBiTree<E>::getElementPriorPointer(Node <E>*& T)
{
    Node <E>* p;
    E min = 0, max = -9999;
    getPriorElement(T, min, max);
    search(T, p, max);
    return p;
}
template<class E>
void BalanceBiTree<E>::getNextElement(Node <E>*& T, E& min, E& max)
{
    if (T)
    {
        max = T->data;
        if (min > max)
            min = max;
        getNextElement(T->LCh, min, max);
        getNextElement(T->RCh, min, max);
    }
}
template<class E>
Node <E>* BalanceBiTree<E>::getElementNextPointer(Node <E>*& T)
{
    Node <E>* p;
    E min = 9999, max = 0;
    getNextElement(T, min, max);
    search(T, p, min);
    return p;
}
template<class E>
void BalanceBiTree<E>::deleteLeafNode(Node <E>*& T, Node <E>*& p, Node <E>*& f)
{
    if (p == NULL)
    {
        cout << "此节点不存在，不能删除" << endl;
        return;
    }
    if (T == p)        //根节点即为叶子节点
    {
        delete p;
        T = NULL;
    }
    else {           //删除节点为非根节点的叶子节点
        if (f->LCh == p)
        {
            delete p;
            f->LCh = NULL;
        }
        if (f->RCh == p)
        {
            delete p;
            f->RCh = NULL;
        }
    }
}
template<class E>
void BalanceBiTree<E>::deleteO(Node <E>*& T, Node <E>*& p, Node <E>*& f)
{
    if (p == NULL)
    {
        cout << "此节点不存在，不能删除" << endl;
        return;
    }
    if (T == p)
    {
        if (T->LCh == NULL && T->RCh != NULL)
        {
            T = p->RCh;
            delete p;
        }
        if (T->RCh == NULL && T->LCh != NULL)
        {
            T = p->LCh;
            delete p;
        }
    }
    else {
        if (p->LCh != NULL)
        {
            if (f->LCh == p)
                f->LCh = p->LCh;
            else
                f->RCh = p->LCh;
        }
        if (p->RCh != NULL)
        {
            if (f->LCh == p)
                f->LCh = p->RCh;
            else
                f->RCh = p->RCh;
        }
    }
}
template<class E>
void BalanceBiTree<E>::deleteT(Node <E>*& T, Node <E>*& p)
{
    Node <E>* f, * next, * prior;
    if (p == NULL)
    {
        cout << "此节点不存在，不能删除" << endl;
        return;
    }
    if (p->balanceFctor == 1)                             //p的平衡因子为1时，用p的前驱节点代替p
    {
        prior = getElementPriorPointer(p->LCh);             //获得x的前驱指针
        if (prior->LCh != NULL && prior->RCh == NULL)   //情况一前驱节点只有左孩子
        {
            p->data = prior->data;
            prior->data = prior->LCh->data;
            delete prior->LCh;
            prior->LCh = NULL;
        }
        if (prior->LCh == NULL && prior->RCh == NULL)    //情况二前驱节点为叶子节点
        {
            getElementFatherPointer(T, f, prior->data); //得到前驱节点的父节点
            p->data = prior->data;
            delete prior;
            f->RCh = NULL;
        }
    }
    else if (p->balanceFctor == -1)                             //p的平衡因子为-1时，用p的后继节点代替p
    {
        next = getElementNextPointer(p->RCh);                //获得x的后继指针
        cout << next->data;
        int level = 1;
        if (next->RCh != NULL && next->LCh == NULL)      //情况一后继节点只有右孩子
        {
            p->data = next->data;
            next->data = next->RCh->data;
            delete next->RCh;
            next->RCh = NULL;
        }
        else if (next->RCh == NULL && next->LCh == NULL)       //情况二后继节点为叶子节点
        {
            getElementFatherPointer(T, f, next->data);     //得到后继节点的父节点
            p->data = next->data;
            delete next;
            f->LCh = NULL;
        }
    }
    else if (p->balanceFctor == 0)     //p的平衡因子为0时，用p的前驱或后继节点代替p，这里用前驱
    {
        prior = getElementPriorPointer(p->LCh);               //获得x的前驱指针
        if (prior->LCh != NULL && prior->RCh == NULL)     //情况一前驱节点只有左孩子
        {
            p->data = prior->data;
            prior->data = prior->LCh->data;
            delete prior->LCh;
            prior->LCh = NULL;
        }
        if (prior->LCh == NULL && prior->RCh == NULL)      //情况二前驱节点为叶子节点
        {
            getElementFatherPointer(T, f, prior->data);     //得到前驱节点的父节点
            p->data = prior->data;
            delete prior;
            if (p == f)                                      //这块需要特殊记忆，唯独p->balanceFctor==0需要考虑***
                f->LCh = NULL;
            else
                f->RCh = NULL;

        }
    }
}
template<class E>
void BalanceBiTree<E>::alldelete(Node <E>*& T, E x)
{
    Node <E>* f, * p = NULL;
    search(T, p, x);
    getElementFatherPointer(T, f, x);
    if (p == NULL)
    {
        cout << "不存在此节点，删除失败！" << endl;
        return;
    }
    if (p->LCh == NULL && p->RCh == NULL)  //情况一删除节点为叶子节点
    {
        deleteLeafNode(T, p, f);
        if (T != NULL)
            AllAdjust(T);
    }
    else if ((p->LCh == NULL && p->RCh != NULL) || (p->LCh != NULL && p->RCh == NULL))
    {
        deleteO(T, p, f);
        if (T != NULL)
            AllAdjust(T);
    }
    else                           //if(p->LCh!=NULL&&p->RCh!=NULL)
    {
        deleteT(T, p);
        if (T != NULL)
            AllAdjust(T);
    }
}
void initArray(int A[])
{
    int i = 0;
    for (i; i < 100; i++)
        A[i] = 0;
}
int main()
{
    int x, y;
    int i = 1;
    int level = 1;
    int A[100] = { 0 };
    int B[100] = { 0 };
    int length = 0;       //存储数组A的有效元素个数
    Node<int>* root;
    Node<int>* p;
    BalanceBiTree<int> T(root);
    BalanceBiTree<int>::menu();
    cout << "请输入执行序号：";
    cin >> x;
    while (x != 0)
    {
        switch (x)
        {
        case 1:
            T.createSubBalanceBiTree(root);
            break;
        case 2:
            cout << "请输入要查询元素的值：";
            cin >> x;
            T.search(root, p, x);
            if (p != NULL)
            {
                if (p->data == x)
                    cout << "元素存在！" << endl;
                else
                    cout << "元素不存在！" << endl;
            }
            else {
                cout << "元素不存在！" << endl;
            }
            break;
        case 3:
            i = 1;
            initArray(A);
            level = 1;
            cout << "请输入要删除元素的值：";
            cin >> x;
            T.alldelete(root, x);
            T.BiTreeToArray(root, A, i, length);
            cout << "其树状图为：" << endl;
            T.LevelTraverse(root, A, length);
            break;
        case 4:
            i = 1;
            initArray(A);
            T.AllAdjust(root);
            T.BiTreeToArray(root, A, i, length);
            cout << "其树状图为：" << endl;
            T.LevelTraverse(root, A, length);
            break;
        }
        system("pause");
        system("CLS");
        BalanceBiTree<int>::menu();
        cout << "请输入执行序号：";
        cin >> x;
    }
    if (root != NULL)
        T.destory(root);
    return 0;
}