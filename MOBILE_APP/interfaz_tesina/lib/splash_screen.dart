import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:interfaz_tesina/ui/web_viewer.dart';

class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  @override

  void initState() {
    super.initState();
    // Add a delay using a Future to simulate loading time
    Future.delayed(const Duration(seconds: 6), () {
      // Navigate to the next page after the delay
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => const WebViewApp(), // Replace NextPage with your actual page
        ),
      );
    });
  }

  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        actions: [],
        systemOverlayStyle: const SystemUiOverlayStyle(
          statusBarBrightness: Brightness.dark, 
          statusBarColor: Colors.black, 
          systemNavigationBarColor: Colors.black
        ),
        backgroundColor: Colors.white,
        elevation: 0,
      ),
      body: Center(
        child: Image.asset('assets/splash_screen.png') 
      
      ,)
    );
  }
}