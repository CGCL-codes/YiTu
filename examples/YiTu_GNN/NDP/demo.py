import sys
import YiTu_GP

print(sys.argv)

YiTu_GNNIndex = sys.argv.index("--YiTu_GNN")

if sys.argv[YiTu_GNNIndex + 1] == "0":
    del sys.argv[YiTu_GNNIndex]
    del sys.argv[YiTu_GNNIndex]
    methodIndex = sys.argv.index("--method")
    a = sys.argv[methodIndex + 1].lower()
    del sys.argv[methodIndex]
    del sys.argv[methodIndex]
    YiTu_GP.YiTu_GP(a, sys.argv)
elif sys.argv[YiTu_GNNIndex + 1] == "1":
    del sys.argv[YiTu_GNNIndex]
    del sys.argv[YiTu_GNNIndex]
    methodIndex = sys.argv.index("--method")
    a = sys.argv[methodIndex + 1].lower()
    del sys.argv[methodIndex]
    del sys.argv[methodIndex]
    del sys.argv[0]
    YiTu_GP.YiTu_GNN(a, ' '.join(sys.argv))
else:
    print("argument value error, YiTu_GNN can only be 0 or 1!")