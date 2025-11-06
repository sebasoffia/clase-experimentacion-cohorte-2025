const localtunnel = require('localtunnel');

(async () => {
  try {
    const tunnel = await localtunnel({ port: 8000 });

    console.log('\n========================================');
    console.log('🌐 TÚNEL PÚBLICO ACTIVO');
    console.log('========================================');
    console.log('URL Pública:', tunnel.url);
    console.log('========================================');
    console.log('\nComparte esta URL con tus alumnos:');
    console.log('📱 Landing simplificada:', tunnel.url + '/landing-simplificada.html');
    console.log('📄 Landing original:', tunnel.url + '/landing.html');
    console.log('========================================\n');
    console.log('⚠️  El túnel permanecerá activo mientras este proceso esté corriendo.');
    console.log('    Presiona Ctrl+C para detenerlo.\n');

    tunnel.on('close', () => {
      console.log('Túnel cerrado');
      process.exit(0);
    });

    // Mantener el proceso vivo
    process.on('SIGINT', () => {
      console.log('\nCerrando túnel...');
      tunnel.close();
    });

  } catch (err) {
    console.error('Error al crear el túnel:', err);
    process.exit(1);
  }
})();
