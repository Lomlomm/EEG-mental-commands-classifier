import 'package:flutter/material.dart';
import 'package:interfaz_tesina/splash_screen.dart';
void main() {
  runApp(
    MaterialApp(
      theme: ThemeData(useMaterial3: true),
      home: const SplashScreen(),
    ),
  );
}