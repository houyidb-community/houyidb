import com.android.build.gradle.LibraryExtension
import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.dsl.KotlinAndroidProjectExtension

// Android library 工程：使用 buildscript classpath（而非 plugins DSL）以便
// 离线缓存解析；AGP 8.5.2 + Kotlin 2.4.21 与 pamobile 宿主工程一致。
// 因为不走 plugins DSL，Kotlin DSL 的 android/implementation 访问器不会生成，
// 这里用显式 configure/add 写法。
// EN: Android library project using a buildscript classpath (instead of the
// plugins DSL) so cached artifacts resolve; AGP 8.5.2 + Kotlin 2.4.21 match the
// pamobile host. Without the plugins DSL, Kotlin DSL accessors for
// android/implementation are not generated, so this uses explicit
// configure/add calls.
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:8.5.2")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.21")
    }
}

apply(plugin = "com.android.library")
apply(plugin = "kotlin-android")

configure<LibraryExtension> {
    namespace = "cc.houyidb.mobile"
    compileSdk = 34
    defaultConfig {
        minSdk = 26
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    defaultConfig {
        consumerProguardFiles("consumer-rules.pro")
    }
}

configure<KotlinAndroidProjectExtension> {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    add("implementation", "org.jetbrains.kotlin:kotlin-stdlib:2.4.21")
}
