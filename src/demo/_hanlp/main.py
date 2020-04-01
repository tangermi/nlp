# -*- coding:utf-8 -*-
import jpype

if __name__=='__main__':
    # 获取系统的jvm路径
    jvm_path = jpype.getDefaultJVMPath()
    print(jvm_path)
    # # 设置jvm路径，以启动java虚拟机
    jpype.startJVM(jvm_path)
    # 执行java代码
    jpype.java.lang.System.out.println('hello world')
    # 关闭jvm虚拟机，当使用完 JVM 后，可以通过 jpype.shutdownJVM() 来关闭 JVM，该函数没有输入参数。当 python 程序退出时，JVM 会自动关闭。
    jpype.shutdownJVM()


