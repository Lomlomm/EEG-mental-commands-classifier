import 'package:flutter/material.dart';
import 'package:dio/dio.dart';

class DataServiceProvider extends StatefulWidget {
  @override
  _DataServiceProviderState createState() => _DataServiceProviderState();
}

class _DataServiceProviderState extends State<DataServiceProvider> {
  late Dio _dio;

  @override
  void initState() {
    super.initState();
    _dio = Dio(); // Aquí puedes configurar el Dio según tus necesidades
  }

  Future<void> fetchProcessData() async {
    try {
      // Aquí colocarías la URL de tu API y los parámetros necesarios
      Response response = await _dio.get('http://127.0.0.1:5000/processData');

      // Aquí manejarías la respuesta, por ejemplo, actualizando el estado del widget
      setState(() {
        // Actualiza el estado del widget con los datos obtenidos
      });
    } catch (error) {
      // Manejar errores, por ejemplo, mostrar un mensaje al usuario
      print('Error al obtener los datos: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('My Service Provider'),
      ),
      body: Center(
        child: ElevatedButton(
          onPressed: fetchProcessData,
          child: Text('Obtener Datos'),
        ),
      ),
    );
  }
}

void main() {
  runApp(MaterialApp(
    title: 'Service Provider Example',
    home: DataServiceProvider(),
  ));
}
