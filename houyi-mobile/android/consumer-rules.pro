# JNI 符号按类/方法名绑定；R8 只保留 native 方法与入口类，避免 release
# 构建把符号重命名或裁掉。
# EN: JNI symbols bind by class/method name; keep the native methods and facade
# so R8 cannot rename or strip them in release builds.
-keep class cc.houyidb.mobile.HouyiDB { *; }
-keepclasseswithmembernames class cc.houyidb.mobile.HouyiDB { native <methods>; }
