# Entrega Final Inteligencia Artificial
**Integrantes**:
- Juan Diego Torres
- Christian Mora
- Jorge Ivan Aguirre

## Datos
Se trabaja con la base de datos WeanDB, que
contiene información sobre pacientes asistidos
por ventilación mecánica.

La base de datos originalmente contaba con **263
pacientes**, de los cuales se tiene la siguiente
información:

### Datos Demográficos
- código del paciente
- hospital de procedencia
- género del paciente, H o M
- edad del paciente

### Datos Clínicos
- **test**: clasificación de los pacientes
    - éxito durante la prueba (grupo 1)
    - no respiración espontánea, reconexión tras 30 minutos (grupo 2)
    - éxito, extubación y respiración espontanea tras 48 horas (grupo 3)
    - éxito durante 30 minutos, pero con reintubación luego de 48 horas (grupo 4)
- **clasificación test**: clasificación extra para los pacientes ya clasificados en el dato anterior (es la clase de la base de datos), se divide en:
    - éxito, que comprende los grupos 1 y 3 de test (S)
    - fracaso, que comprende el grupo 2 de test (F)
    - reintubado, es el grupo 4 de test (R)
- **Modo**: modo en el que fue ventilado el paciente, puede ser:
    - presión soporte (PS)
    - asistido controlado (AC)

### Datos Cardíacos
- **TAS_antes**: presión arterial sistólica antes de intubar
- **TAD_antes**: presión arterial diastólica antes de intubar
- **FC_antes**: frecuencia cardíaca antes de intubar
- **TAS**: presión arterial sistólica después de intubar
- **TAD**: presión arterial diastólica después de intubar
- **FC**: frecuencia cardíaca después de intubar

### Datos Respiratorios
- **FR_antes**: frecuencia respiratoria antes de intubar
- **VT_antes**: volumen tidal antes de intubar
- **PEEP**: positive end respiratory pressure antes de intubar
- **FiO2_antes**: fracción inspirada de oxígeno antes de la prueba
- **FR**: frecuencia respiratoria después de intubar
- **VT**: volumen tidal después de intubar
- **FiO2**: fracción inspirada de oxígeno después de la prueba

### Otros Valores Característicos
- **TEMP**: temperatura del paciente
- **dVM**: días de ventilación mecánica
- **Hb**: hemoglobina

### Diagnósticos
- **Diagnóstico Principal (Dpr)**: diagnóstico principal que dio el doctor, se clasificó en:
    - ICC: insuficiencia cardíaca
    - NRL: enfermedad neurológica
    - PULM: enfermedad pulmonar
    - PABD: patología abdominal
    - PSTC: postoperatorio cirugía cardíaca
    - MISC: miscelánea
- **Diagnostico Insuficiencia Respiratoria Aguda (DIRA)**: diagnóstico que dio el doctor sobre la enfermedad respiratoria que tiene el paciente, si la tiene, se clasificó en:
    - ICC: insuficiencia cardíaca
    - NRL: enfermedad neurológica
    - PULM: enfermedad pulmonar
    - PABD: patología abdominal
    - MISC: miscelánea
    - NONE: no posee IRA
- **Causa de la IRA**: razón por la cual el paciente tiene la insuficiencia respiratoria aguda, si es que la tiene, son:
    - PULM1: pulmonar primario como EPOC o pulmonía
    - PULM2: pulmonar secundario como ICC o patologías abdominales
    - NRL: paciente neurológico 
    - NONE: no posee IRA

### Tratamiento de los Datos
para hacer una mejor predicción con el clasificador a usar se modificó la base de datos original.
Se removieron las columnas:
- código, ya que no ayuda a la predicción
- hospital, ya que no ayuda a la predicción
- Test, debido a que es una versión más especifica de la clase
- Diagnóstico Principal, debido a que existe la columna **Código DPr** que tiene la información resumida
- Diagnóstico Principal 2, debido a que existe la columna **Código DPr** que tiene la información resumida 
- Diagnóstico Principal 3, debido a que existe la columna **Código DPr** que tiene la información resumida
- Clasificación DPr, debido a que existe la columna **Código DPr** que tiene la información resumida
- Diagnóstico IRA, debido a que existe la columna **Código DIRA** que tiene la información resumida
- Diagnóstico IRA 1, debido a que existe la columna **Código DIRA** que tiene la información resumida
- Diagnóstico IRA 2, debido a que existe la columna **Código DIRA** que tiene la información resumida
- Clasificación DIRA, debido a que existe la columna **Código DIRA** que tiene la información resumida
- Clasificación PULM, debido a que existe la columna **Código PULM** que tiene la información resumida
- Modo, debido a que existe la columna **Modo abr** que tiene la información resumida

Luego se renombraron y reorganizaron las columnas a como se describen en la sección de datos.
Adicionalmente se eliminaron las filas que no contenían información completa sobre el paciente o contaban con clase como es el caso del paciente con id p0010.

### Bases de Datos
- [db original](./data/weandb.ods)
- [db con pacientes incompletos removidos](./data/weandbFiltrado.ods)