# Desarrollo de un sistema de diagnóstico de enfermedades y clasificación de especies en hojas de plantas mediante PlantCV y modelos de aprendizaje profundo

Descripcion: La detección temprana de enfermedades en cultivos es vital para la agricultura sostenible. Utilizando PlantCV (https://plantcv.readthedocs.io/), una biblioteca enfocada en la visión por computadora para plantas, junto con TensorFlow o PyTorch, el alumno se encargará de crear modelos de clasificación multiclase robustos. Estos modelos tendrán como objetivo identificar distintas enfermedades y una clase de salud en hojas de varias especies. Para el entrenamiento y evaluación, se usará el New Plant Diseases Dataset de Kaggle (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset), que contiene más de 80,000 imágenes recogidas en entornos de laboratorio e in situ. Dada la naturaleza práctica de la aplicación, se buscará desarrollar un modelo lo suficientemente eficiente como para poder ser implementado en aplicaciones móviles, permitiendo diagnósticos ágiles y efectivos en el campo.



## Instalación

Para ejecutar los servicios usando docker-compose:

```bash
$ cp .env.original .env
```

```bash
$ docker network create shared_network
```

Solo para usuarios de Mac con procesador M1:
- Hay un Dockerfile específico para Mac M1 en model/Dockerfile.M1. Este Dockerfile descarga TensorFlow    compilado para M1.
- Modifica docker-compose.yaml para que use ese Dockerfile.
- Elimina TensorFlow de requirements.txt
- Recuerda restaurar docker-compose.yaml y requirements.txt antes de hacer la entrega.

**Advertencia:** No podrás iniciar el proyecto hasta completar el Dockerfile que se encuentra en la carpeta api, como se menciona en el archivo ASSIGNMENT.md.

```bash
$ docker-compose up --build -d
```

Para detener los servicios:

```bash
$ docker-compose down
```

Poblar la base de datos:
```bash
cd api
cp .env.original .env
docker-compose up --build -d
```


## Acceder a la documentación de FastAPI

URL = http://localhost:8000/docs


![Sample Image](fastapi_docs.png)

Para probar los endpoints, necesitas autenticarte con: user = admin@example.com password = admin


## Acceder a la Interfaz de Usuario

URL = http://localhost:9090

![Sample Image](ui_login.png)

![Sample Image](ui_classify.png)

- Inicia sesión con:
    - user: `admin@example.com`
    - pass: `admin`
- Puedes subir una imagen.
- Puedes clasificarla.


