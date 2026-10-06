/*
  SMART GREENHOUSE AUTOMATION SYSTEM (SGAS)
  Group Project
*/

#include <DHT.h>
#include <ESP32Servo.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"

// THINGSPEAK CONFIGURATION
#include "secrets.h"
const char* THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update";

const unsigned long THINGSPEAK_UPLOAD_INTERVAL = 20000;
const unsigned long FIRST_UPLOAD_GRACE = 10000;  // First attempt: 10 s after Wi-Fi setup, once ready.
unsigned long lastThingSpeakUpload = 0;
unsigned long networkStartTime = 0;
bool thingSpeakUploadAttempted = false;

// PIN CONFIGURATION

const int DHT_PIN = 15;
const int LDR_PIN = 34;

const int SERVO_PIN = 18;

const int GROW_LIGHT_PIN = 13; // Green LED
const int HEATER_PIN = 26;     // Red LED
const int HUMIDIFIER_PIN = 27; // Blue LED
const int BUZZER_PIN = 25;

// SENSOR AND SERVO SETUP

#define DHT_TYPE DHT22
DHT dht(DHT_PIN, DHT_TYPE);

Servo ventServo;

const int VENT_CLOSED_ANGLE = 0;
const int VENT_OPEN_ANGLE = 90;

// ENVIRONMENTAL THRESHOLDS

// Temperature hysteresis
const float HOT_ENTER = 30.0;
const float HOT_RECOVER = 27.0;

const float COLD_ENTER = 18.0;
const float COLD_RECOVER = 20.0;

// Humidity hysteresis
const float HUMID_ENTER = 80.0;
const float HUMID_RECOVER = 70.0;

const float DRY_ENTER = 40.0;
const float DRY_RECOVER = 50.0;

// Light hysteresis
const int DARK_ENTER = 2500;
const int DARK_RECOVER = 1800;

// FINITE STATE MACHINE

enum GreenhouseState
{
  NORMAL,
  HOT,
  COLD,
  HUMID,
  DRY,
  LOW_LIGHT,
  ALERT
};

GreenhouseState currentState = NORMAL;

// SENSOR VALUES

float temperature = 0.0;
float humidity = 0.0;
int lightLevel = 0;

bool haveValidDHTReading = false;
bool sensorFault = false;

// ACTVE ENVIRONMENTAL CONDITIONS
// Multiple conditions can be true at the same time, e.g. COLD + LOW_LIGHT
bool hotCondition = false;
bool coldCondition = false;
bool humidCondition = false;
bool dryCondition = false;
bool lowLightCondition = false;

// ACTUATOR STATES
bool ventOpen = false;
bool heaterOn = false;
bool humidifierOn = false;
bool growLightOn = false;

// NON-BLOCKING / ADAPTIVE TIMING
// Local control is slower during NORMAL operation and faster when an abnormal condition is active
const unsigned long NORMAL_POLL_INTERVAL = 2000;
const unsigned long FAST_POLL_INTERVAL = 500;

// DHT22 should not be physically read every 500 ms.
const unsigned long DHT_READ_INTERVAL = 2000;

unsigned long controlInterval = NORMAL_POLL_INTERVAL;
unsigned long lastControlTime = 0;
unsigned long lastDHTTime = 0;


// WIFI AND MQTT CONFIGURATION
const char *WIFI_SSID = "Wokwi-GUEST";
const char *WIFI_PASSWORD = "";

const char *MQTT_BROKER = "broker.hivemq.com";
const uint16_t MQTT_PORT = 1883;
const char *MQTT_TELEMETRY_TOPIC = "sgas/your-name/telemetry";
const char *MQTT_COMMAND_TOPIC = "sgas/your-name/command";
const char *MQTT_CLASSIFICATION_TOPIC = "sgas/your-name/classification";

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);

const unsigned long WIFI_RETRY_INTERVAL = 5000;
const unsigned long MQTT_RETRY_INTERVAL = 5000;
const unsigned long MQTT_PUBLISH_INTERVAL = 2000;

unsigned long lastWiFiRetry = 0;
unsigned long lastMQTTRetry = 0;
unsigned long lastMQTTPublish = 0;
bool wifiWasConnected = false;
bool mqttWasConnected = false;

// TINY ML SETUP

const tflite::Model *tflModel = nullptr;
tflite::MicroInterpreter *tflInterpreter = nullptr;

TfLiteTensor *tflInput = nullptr;
TfLiteTensor *tflOutput = nullptr;

bool tinyMLReady = false;

// Memory used while the neural network is running.
// We can tune this later.
constexpr size_t TENSOR_ARENA_SIZE = 100 * 1024;

uint8_t *tensorArena = nullptr;

bool classifyRequested = false;
char classificationLeafId[17] = "";

// FUNCTION DECLARATIONS

void setupHardware();
void setupWiFi();
void setupMQTT();
void maintainWiFi();
void maintainMQTT();
String createMQTTClientId();
void mqttCallback(char *topic, byte *payload, unsigned int length);
void publishMQTT();
void uploadThingSpeak();
void publishClassification(const char* status, const char* error, float probability);
void runLocalAutomation(unsigned long currentTime);
bool setupTinyML();


// HELPERS

uint16_t readLE16(const uint8_t* data)
{
  return (uint16_t)data[0]
       | ((uint16_t)data[1] << 8);
}

uint32_t readLE32(const uint8_t* data)
{
  return (uint32_t)data[0]
       | ((uint32_t)data[1] << 8)
       | ((uint32_t)data[2] << 16)
       | ((uint32_t)data[3] << 24);
}

// SETUP

void setup()
{
  Serial.begin(115200);

  tinyMLReady = setupTinyML();
  if (tinyMLReady)
  {
    Serial.println("TinyML: READY");
  }
  else
  {
    Serial.println("TinyML: FAILED");
  }
  setupHardware();
  setupWiFi();
  setupMQTT();
}

bool setupTinyML()
{

  Serial.println();
  Serial.println("Initializing TinyML model...");

  Serial.print("Total free heap: ");
  Serial.println(ESP.getFreeHeap());

  Serial.print("Largest allocatable block: ");
  Serial.println(ESP.getMaxAllocHeap());

  tensorArena = (uint8_t *)malloc(TENSOR_ARENA_SIZE);

  if (tensorArena == nullptr)
  {
    Serial.println("ERROR: Could not allocate tensor arena.");
    Serial.print("Free heap: ");
    Serial.println(ESP.getFreeHeap());
    return false;
  }

  Serial.print("Tensor arena allocated: ");
  Serial.print(TENSOR_ARENA_SIZE / 1024);
  Serial.println(" KB");

  Serial.print("Free heap remaining: ");
  Serial.println(ESP.getFreeHeap());

  tflite::InitializeTarget();

  tflModel = tflite::GetModel(model_data);

  if (tflModel->version() != TFLITE_SCHEMA_VERSION)
  {
    Serial.println("ERROR: TFLite schema version mismatch.");
    return false;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter staticInterpreter(
      tflModel,
      resolver,
      tensorArena,
      TENSOR_ARENA_SIZE);

  tflInterpreter = &staticInterpreter;

  if (tflInterpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("ERROR: AllocateTensors() failed.");
    return false;
  }

  Serial.print("Tensor arena used: ");
  Serial.print(tflInterpreter->arena_used_bytes() / 1024.0);
  Serial.println(" KB");

  tflInput = tflInterpreter->input(0);
  tflOutput = tflInterpreter->output(0);

  if (tflInput->type != kTfLiteInt8 || tflInput->dims->size != 4 ||
      tflInput->dims->data[0] != 1 || tflInput->dims->data[1] != 64 ||
      tflInput->dims->data[2] != 64 || tflInput->dims->data[3] != 3 ||
      tflInput->bytes < 64 * 64 * 3 || tflInput->params.scale <= 0 ||
      tflOutput->type != kTfLiteInt8 || tflOutput->bytes != 1 ||
      tflOutput->params.scale <= 0)
  {
    Serial.println("ERROR: Expected INT8 [1,64,64,3] input and one INT8 probability output.");
    return false;
  }

  Serial.println("TinyML model loaded successfully.");

  return true;
}

bool loadBMPFromHTTPToTensor(const char* url)
{
  if (WiFi.status() != WL_CONNECTED)
  {
    Serial.println("ERROR: Wi-Fi not connected.");
    return false;
  }

  HTTPClient http;

  Serial.print("Downloading image: ");
  Serial.println(url);

  http.begin(url);

  int httpCode = http.GET();

  if (httpCode != HTTP_CODE_OK)
  {
    Serial.print("ERROR: HTTP response ");
    Serial.println(httpCode);

    http.end();
    return false;
  }

  Serial.print("Image size: ");
  Serial.print(http.getSize());
  Serial.println(" bytes");

  WiFiClient* stream = http.getStreamPtr();
  stream->setTimeout(5000);

  // Standard BMP header
  uint8_t header[54];

  if (stream->readBytes(header, 54) != 54)
  {
    Serial.println("ERROR: Could not read BMP header.");

    http.end();
    return false;
  }

  // BMP must start with BM
  if (header[0] != 'B' || header[1] != 'M')
  {
    Serial.println("ERROR: Invalid BMP file.");

    http.end();
    return false;
  }

  uint32_t pixelOffset = readLE32(&header[10]);

  int32_t width = (int32_t)readLE32(&header[18]);

  int32_t height = (int32_t)readLE32(&header[22]);

  uint16_t planes = readLE16(&header[26]);

  uint16_t bitDepth = readLE16(&header[28]);

  uint32_t compression = readLE32(&header[30]);

  Serial.print("BMP dimensions: ");
  Serial.print(width);
  Serial.print(" x ");
  Serial.println(height);

  if (width != 64 || abs(height) != 64)
  {
    Serial.println("ERROR: Image must be 64x64.");

    http.end();
    return false;
  }

  if (planes != 1 ||
      bitDepth != 24 ||
      compression != 0)
  {
    Serial.println("ERROR: Image must be uncompressed 24-bit BMP.");

    http.end();
    return false;
  }

  // Skip anything between the 54-byte header
  // and the actual pixel data.
  if (pixelOffset < 54)
  {
    Serial.println("ERROR: Invalid BMP pixel offset.");

    http.end();
    return false;
  }

  uint32_t extraBytes = pixelOffset - 54;

  if (extraBytes > 4096)
  {
    Serial.println("ERROR: Unsupported BMP pixel offset.");
    http.end();
    return false;
  }
  while (extraBytes > 0)
  {
    uint8_t ignored;
    if (stream->readBytes(&ignored, 1) != 1)
    {
      Serial.println("ERROR: Truncated BMP metadata.");
      http.end();
      return false;
    }
    extraBytes--;
  }

  // BMP rows are padded to multiples of 4 bytes.
  uint32_t rowSize = (width * 3 + 3) & ~3;

  uint32_t padding = rowSize - (width * 3);

  bool topDown = height < 0;

  float inputScale = tflInput->params.scale;

  int inputZeroPoint = tflInput->params.zero_point;

  uint8_t pixel[3];

  for (int row = 0; row < 64; row++)
  {
    // Standard BMP files store rows bottom-up.
    int targetY = topDown ? row : 63 - row;

    for (int x = 0; x < 64; x++)
    {
      if (stream->readBytes(pixel, 3) != 3)
      {
        Serial.println("ERROR: Unexpected end of image.");
        http.end();
        return false;
      }

      // BMP order = B G R
      uint8_t blue  = pixel[0];
      uint8_t green = pixel[1];
      uint8_t red   = pixel[2];

      // Normalize to [0,1] range
      float r = red   / 255.0f;
      float g = green / 255.0f;
      float b = blue  / 255.0f;

      // Float -> INT8
      int qr = round(r / inputScale) + inputZeroPoint;
      int qg = round(g / inputScale) + inputZeroPoint;
      int qb = round(b / inputScale) + inputZeroPoint;

      qr = constrain(qr, -128, 127);
      qg = constrain(qg, -128, 127);
      qb = constrain(qb, -128, 127);

      int index = (targetY * 64 + x) * 3;

      tflInput->data.int8[index] = (int8_t)qr;
      tflInput->data.int8[index + 1] = (int8_t)qg;
      tflInput->data.int8[index + 2] = (int8_t)qb;
    }

    // Skip BMP row padding.
    for (uint32_t i = 0; i < padding; i++)
    {
      stream->read();
    }
  }

  http.end();

  Serial.println("Image loaded into TinyML tensor.");

  return true;
}

void runPlantDiseaseInference(const char* url)
{
  publishClassification("running", "", 0);
  if (!tinyMLReady)
  {
    Serial.println("TinyML model not ready.");
    publishClassification("error", "TinyML model not ready. Check ESP32 startup logs.", 0);
    return;
  }

  Serial.println();
  Serial.println(
    "===== TinyML Plant Disease Test ====="
  );

  if (!loadBMPFromHTTPToTensor(url))
  {
    Serial.println("Image loading failed.");
    publishClassification("error", "Image download failed. Check leaf server and Wokwi network access.", 0);
    return;
  }

  Serial.println("Running inference...");

  if (tflInterpreter->Invoke() != kTfLiteOk)
  {
    Serial.println("ERROR: TinyML inference failed.");
    publishClassification("error", "TinyML inference failed. Check Serial Monitor.", 0);
    return;
  }

  int rawOutput = static_cast<int>(tflOutput->data.int8[0]);
  float probability = (rawOutput - tflOutput->params.zero_point) * tflOutput->params.scale;

  if (!isfinite(probability))
  {
    publishClassification("error", "Model returned an invalid probability.", 0);
    return;
  }
  probability = constrain(probability, 0.0f, 1.0f);
  publishClassification("done", "", probability);

  Serial.print("Disease probability: ");
  Serial.println(probability, 4);

  if (probability >= 0.5)
  {
    Serial.println("Prediction: DISEASED");
  }
  else
  {
    Serial.println("Prediction: HEALTHY");
  }

  Serial.println(
    "====================================="
  );
}

void publishClassification(const char* status, const char* error, float probability)
{
  // IDs are validated hex strings; status and error are fixed strings below.
  char result[256];
  if (strcmp(status, "done") == 0)
  {
    snprintf(result, sizeof(result),
      "{\"leafId\":\"%s\",\"status\":\"done\",\"prediction\":\"%s\",\"probability\":%.4f}",
      classificationLeafId, probability >= 0.5f ? "DISEASED" : "HEALTHY", probability);
  }
  else
  {
    snprintf(result, sizeof(result),
      "{\"leafId\":\"%s\",\"status\":\"%s\",\"error\":\"%s\"}",
      classificationLeafId, status, error);
  }
  Serial.println(result);
  if (!mqttClient.publish(MQTT_CLASSIFICATION_TOPIC, result, false))
  {
    Serial.println("Could not publish classification status; check MQTT connection.");
  }
}

void setupHardware()
{
  // Configure outputs
  pinMode(GROW_LIGHT_PIN, OUTPUT);
  pinMode(HEATER_PIN, OUTPUT);
  pinMode(HUMIDIFIER_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Configure LDR input
  pinMode(LDR_PIN, INPUT);

  // Start all outputs OFF
  digitalWrite(GROW_LIGHT_PIN, LOW);
  digitalWrite(HEATER_PIN, LOW);
  digitalWrite(HUMIDIFIER_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);

  // Start DHT22
  dht.begin();

  // Start servo and close ventilation
  ventServo.setPeriodHertz(50);
  ventServo.attach(SERVO_PIN, 500, 2400);
  ventServo.write(VENT_CLOSED_ANGLE);

  // Initial light reading
  lightLevel = analogRead(LDR_PIN);

  Serial.println();
  Serial.println("==================================");
  Serial.println(" SGAS - Smart Greenhouse System");
  Serial.println(" Local automation started");
  Serial.println("==================================");
}

// MAIN LOOP

void loop()
{
  unsigned long currentTime = millis();

  maintainWiFi();
  maintainMQTT();
  runLocalAutomation(currentTime);

  if (classifyRequested && WiFi.status() == WL_CONNECTED)
  {
    classifyRequested = false;
    char imageURL[80];
    snprintf(imageURL, sizeof(imageURL),
      "http://host.wokwi.internal:8000/leaves/%s.bmp", classificationLeafId);
    runPlantDiseaseInference(imageURL);
  }

  // Independent telemetry timer; stays outside the local-control if block.
  currentTime = millis();  // Inference and connection attempts may have taken time.
  if (currentTime - lastMQTTPublish >= MQTT_PUBLISH_INTERVAL)
  {
    lastMQTTPublish = currentTime;
    publishMQTT();
  }

  const bool thingSpeakDue = thingSpeakUploadAttempted
    ? currentTime - lastThingSpeakUpload >= THINGSPEAK_UPLOAD_INTERVAL
    : currentTime - networkStartTime >= FIRST_UPLOAD_GRACE;

  if (thingSpeakDue && WiFi.status() == WL_CONNECTED && haveValidDHTReading && !sensorFault)
  {
    uploadThingSpeak();
    thingSpeakUploadAttempted = true;
    lastThingSpeakUpload = millis();  // Keep attempts spaced even after slow requests.
    maintainMQTT();
  }
}

void runLocalAutomation(unsigned long currentTime)
{
  // Read DHT22 at a safe interval.
  if (!haveValidDHTReading ||
      currentTime - lastDHTTime >= DHT_READ_INTERVAL)
  {
    lastDHTTime = currentTime;
    readDHT22();
  }

  // Run the main Sense, Think, Act cycle
  if (currentTime - lastControlTime >= controlInterval)
  {
    lastControlTime = currentTime;

    // SENSE
    readLightSensor();

    // THINK
    // Update LDR conditions even if the DHT22 has failed.
    updateEnvironmentalConditions();
    if (!sensorFault)
    {
      determinePrimaryState();
    }
    else
    {
      currentState = ALERT;
    }

    // ACT
    updateActuators();

    // Change control frequency depending on system condition.
    updatePollingInterval();

    // Local monitoring
    printSystemStatus();
  }
}

// SENSOR READING

void readDHT22()
{
  float newTemperature = dht.readTemperature();
  float newHumidity = dht.readHumidity();

  // Detect NaN or unrealistic sensor values.
  if (isnan(newTemperature) ||
      isnan(newHumidity) ||
      newTemperature < -40.0 ||
      newTemperature > 80.0 ||
      newHumidity < 0.0 ||
      newHumidity > 100.0)
  {

    sensorFault = true;
    Serial.println("WARNING: DHT22 sensor fault.");
    return;
  }

  temperature = newTemperature;
  humidity = newHumidity;

  haveValidDHTReading = true;
  sensorFault = false;
}

void readLightSensor()
{
  lightLevel = analogRead(LDR_PIN);
}

// HYSTERESIS / CONDITION DETECTION

void updateEnvironmentalConditions()
{
  // Temperature and humidity must not use readings from a failed DHT22.
  if (!sensorFault)
  {
    // HOT: enter >30 C, recover <27 C
    if (!hotCondition && temperature > HOT_ENTER)
    {
      hotCondition = true;
    }
    else if (hotCondition && temperature < HOT_RECOVER)
    {
      hotCondition = false;
    }

    // COLD: enter <18 C, recover >20 C
    if (!coldCondition && temperature < COLD_ENTER)
    {
      coldCondition = true;
    }
    else if (coldCondition && temperature > COLD_RECOVER)
    {
      coldCondition = false;
    }

    // HUMID: enter >80%, recover <70%
    if (!humidCondition && humidity > HUMID_ENTER)
    {
      humidCondition = true;
    }
    else if (humidCondition && humidity < HUMID_RECOVER)
    {
      humidCondition = false;
    }

    // DRY: enter <40%, recover >50%
    if (!dryCondition && humidity < DRY_ENTER)
    {
      dryCondition = true;
    }
    else if (dryCondition && humidity > DRY_RECOVER)
    {
      dryCondition = false;
    }
  }

  // The LDR is independent, so it keeps working during a DHT22 fault.
  // LOW LIGHT: uses hysteresis to avoid LED flickering.
  if (!lowLightCondition && lightLevel > DARK_ENTER)
  {
    lowLightCondition = true;
  }
  else if (lowLightCondition && lightLevel < DARK_RECOVER)
  {
    lowLightCondition = false;
  }
}

// PRIMARY FSM STATE
// The system has one primary FSM state for classification,
// while condition flags allow multiple conditions simultaneously.

void determinePrimaryState()
{
  if (sensorFault)
  {
    currentState = ALERT;
  }
  else if (coldCondition)
  {
    currentState = COLD;
  }
  else if (hotCondition)
  {
    currentState = HOT;
  }
  else if (humidCondition)
  {
    currentState = HUMID;
  }
  else if (dryCondition)
  {
    currentState = DRY;
  }
  else if (lowLightCondition)
  {
    currentState = LOW_LIGHT;
  }
  else
  {
    currentState = NORMAL;
  }
}

// ACTUATOR CONTROL

void updateActuators()
{
  // SENSOR FAULT / ALERT
  if (sensorFault)
  {
    // DHT-dependent control is disabled for safety.
    ventOpen = false;
    heaterOn = false;
    humidifierOn = false;

    // LDR is independent, so lighting can continue.
    growLightOn = lowLightCondition;

    digitalWrite(BUZZER_PIN, HIGH);
  }

  // NORMAL SENSOR OPERATION
  else
  {
    digitalWrite(BUZZER_PIN, LOW);

    /*
      Ventilation:
      - COLD has priority and forces the vent closed.
      - Otherwise HOT or HUMID opens the vent.
    */
    if (coldCondition)
    {
      ventOpen = false;
    }
    else
    {
      ventOpen = hotCondition || humidCondition;
    }

    heaterOn = coldCondition;
    humidifierOn = dryCondition;
    growLightOn = lowLightCondition;
  }

  // Apply servo position
  if (ventOpen)
  {
    ventServo.write(VENT_OPEN_ANGLE);
  }
  else
  {
    ventServo.write(VENT_CLOSED_ANGLE);
  }

  // Apply LED actuator states
  digitalWrite(HEATER_PIN, heaterOn ? HIGH : LOW);
  digitalWrite(HUMIDIFIER_PIN, humidifierOn ? HIGH : LOW);
  digitalWrite(GROW_LIGHT_PIN, growLightOn ? HIGH : LOW);
}

// ADAPTIVE POLLING

void updatePollingInterval()
{
  if (currentState == NORMAL)
  {
    controlInterval = NORMAL_POLL_INTERVAL;
  }
  else
  {
    controlInterval = FAST_POLL_INTERVAL;
  }
}

// STATE NAME

const char *stateToString()
{
  switch (currentState)
  {
  case NORMAL:
    return "NORMAL";
  case HOT:
    return "HOT";
  case COLD:
    return "COLD";
  case HUMID:
    return "HUMID";
  case DRY:
    return "DRY";
  case LOW_LIGHT:
    return "LOW LIGHT";
  case ALERT:
    return "ALERT";
  default:
    return "UNKNOWN";
  }
}

// SERIAL MONITOR OUTPUT

void printSystemStatus()
{
  Serial.println();
  Serial.println("========== SGAS STATUS ==========");

  Serial.print("Temperature: ");
  if (haveValidDHTReading && !sensorFault)
  {
    Serial.print(temperature, 1);
    Serial.println(" C");
  }
  else
  {
    Serial.println("NO VALID READING");
  }

  Serial.print("Humidity:    ");
  if (haveValidDHTReading && !sensorFault)
  {
    Serial.print(humidity, 1);
    Serial.println(" %");
  }
  else
  {
    Serial.println("NO VALID READING");
  }

  Serial.print("Light ADC:   ");
  Serial.println(lightLevel);

  Serial.print("FSM State:   ");
  Serial.println(stateToString());

  // Display all active conditions.
  Serial.print("Conditions:  ");

  bool anyCondition = false;

  if (hotCondition)
  {
    Serial.print("HOT ");
    anyCondition = true;
  }

  if (coldCondition)
  {
    Serial.print("COLD ");
    anyCondition = true;
  }

  if (humidCondition)
  {
    Serial.print("HUMID ");
    anyCondition = true;
  }

  if (dryCondition)
  {
    Serial.print("DRY ");
    anyCondition = true;
  }

  if (lowLightCondition)
  {
    Serial.print("LOW_LIGHT ");
    anyCondition = true;
  }

  if (sensorFault)
  {
    Serial.print("FAULT ");
    anyCondition = true;
  }

  if (!anyCondition)
  {
    Serial.print("NONE");
  }

  Serial.println();

  // Actuator status
  Serial.println("---------------------------------");

  Serial.print("Vent:        ");
  Serial.println(ventOpen ? "OPEN" : "CLOSED");

  Serial.print("Heater:      ");
  Serial.println(heaterOn ? "ON" : "OFF");

  Serial.print("Humidifier:  ");
  Serial.println(humidifierOn ? "ON" : "OFF");

  Serial.print("Grow Light:  ");
  Serial.println(growLightOn ? "ON" : "OFF");

  Serial.print("Fault Alarm: ");
  Serial.println(sensorFault ? "ON" : "OFF");

  Serial.print("Control Poll:");
  Serial.print(controlInterval);
  Serial.println(" ms");

  Serial.println("=================================");
}

// WIFI AND MQTT FUNCTIONS

void setupWiFi()
{
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWiFiRetry = millis();
  networkStartTime = lastWiFiRetry;
}

void setupMQTT()
{
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setBufferSize(512);
  mqttClient.setSocketTimeout(2);
  mqttClient.setKeepAlive(60);  // Allow room for HTTPS uploads and image inference.
}

String createMQTTClientId()
{
  // A stable, device-specific ID avoids using the same ID as Node-RED.
  uint64_t chipId = ESP.getEfuseMac();
  char clientId[40];
  snprintf(clientId, sizeof(clientId), "SGAS-%08lX%08lX",
           (unsigned long)(chipId >> 32), (unsigned long)chipId);
  return String(clientId);
}

void maintainWiFi()
{
  if (WiFi.status() == WL_CONNECTED)
  {
    if (!wifiWasConnected)
    {
      wifiWasConnected = true;
      Serial.print("Wi-Fi connected. IP: ");
      Serial.println(WiFi.localIP());
    }
    return;
  }

  if (wifiWasConnected)
  {
    wifiWasConnected = false;
    Serial.println("Wi-Fi disconnected.");
  }

  unsigned long now = millis();
  if (now - lastWiFiRetry >= WIFI_RETRY_INTERVAL)
  {
    lastWiFiRetry = now;
    Serial.println("Retrying Wi-Fi...");
    WiFi.reconnect();
  }
}

void maintainMQTT()
{
  // loop() can detect a lost connection or a keepalive timeout.
  if (WiFi.status() == WL_CONNECTED && mqttClient.connected())
  {
    mqttClient.loop();
  }

  if (mqttWasConnected && (WiFi.status() != WL_CONNECTED || !mqttClient.connected()))
  {
    mqttWasConnected = false;
    Serial.print("MQTT disconnected. State: ");
    Serial.println(mqttClient.state());
  }

  if (WiFi.status() != WL_CONNECTED)
  {
    return;
  }

  if (mqttClient.connected())
  {
    return;
  }

  unsigned long now = millis();
  if (now - lastMQTTRetry < MQTT_RETRY_INTERVAL)
  {
    return;
  }
  lastMQTTRetry = now;

  String clientId = createMQTTClientId();
  Serial.print("Connecting to HiveMQ as ");
  Serial.print(clientId);
  Serial.println("...");

  if (mqttClient.connect(clientId.c_str()))
  {
    mqttWasConnected = true;
    Serial.println("Connected to HiveMQ!");

    if (mqttClient.subscribe(MQTT_COMMAND_TOPIC))
    {
      Serial.print("Subscribed to: ");
      Serial.println(MQTT_COMMAND_TOPIC);
    }
    else
    {
      Serial.println("Command topic subscription failed.");
    }
  }
  else
  {
    Serial.print("MQTT connection failed. State: ");
    Serial.println(mqttClient.state());
  }
}

void mqttCallback(char *topic, byte *payload, unsigned int length)
{
  // Dashboard commands contain CLASSIFY: followed by a 16-character leaf ID.
  if (strcmp(topic, MQTT_COMMAND_TOPIC) != 0 || length != 25 || classifyRequested ||
      memcmp(payload, "CLASSIFY:", 9) != 0)
  {
    return;
  }

  for (unsigned int i = 9; i < length; i++)
  {
    if (!((payload[i] >= '0' && payload[i] <= '9') ||
          (payload[i] >= 'a' && payload[i] <= 'f')))
    {
      return;
    }
  }

  memcpy(classificationLeafId, payload + 9, 16);
  classificationLeafId[16] = '\0';
  classifyRequested = true;
  Serial.print("Leaf classification requested: ");
  Serial.println(classificationLeafId);
}

void publishMQTT()
{
  if (!mqttClient.connected())
  {
    return;
  }

  char tempText[20];
  char humidityText[20];
  if (sensorFault || !haveValidDHTReading)
  {
    snprintf(tempText, sizeof(tempText), "null");
    snprintf(humidityText, sizeof(humidityText), "null");
  }
  else
  {
    snprintf(tempText, sizeof(tempText), "%.1f", temperature);
    snprintf(humidityText, sizeof(humidityText), "%.1f", humidity);
  }

  char payload[384];
  int size = snprintf(payload, sizeof(payload),
                      "{\"temperature\":%s,\"humidity\":%s,\"light\":%d,"
                      "\"state\":\"%s\",\"vent\":%s,\"heater\":%s,"
                      "\"humidifier\":%s,\"growLight\":%s,\"alert\":%s}",
                      tempText, humidityText, lightLevel, stateToString(),
                      ventOpen ? "true" : "false",
                      heaterOn ? "true" : "false",
                      humidifierOn ? "true" : "false",
                      growLightOn ? "true" : "false",
                      sensorFault ? "true" : "false");

  if (size < 0 || size >= (int)sizeof(payload))
  {
    Serial.println("MQTT payload too large.");
    return;
  }

  if (mqttClient.publish(MQTT_TELEMETRY_TOPIC, payload))
  {
    Serial.println("MQTT telemetry published.");
  }
  else
  {
    Serial.println("MQTT telemetry publish failed.");
  }
}

void uploadThingSpeak() {
  Serial.println("ThingSpeak: starting upload...");
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("ThingSpeak: WiFi disconnected");
    return;
  }

  if (!haveValidDHTReading || sensorFault) {
    Serial.println("ThingSpeak: Invalid sensor data");
    return;
  }

  WiFiClientSecure client;
  client.setInsecure();
  client.setHandshakeTimeout(5);  // Seconds; avoid a long pause in MQTT servicing.
  HTTPClient http;

  String url = String(THINGSPEAK_UPDATE_URL) +
                    "?api_key=" + String(THINGSPEAK_WRITE_API_KEY) +
                    "&field1=" + String(temperature, 2) +
                    "&field2=" + String(humidity, 2) +
                    "&field3=" + String(lightLevel) +
                    "&field4=" + String(ventOpen ? 1 : 0) +
                    "&field5=" + String(heaterOn ? 1 : 0) +
                    "&field6=" + String(humidifierOn ? 1 : 0) +
                    "&field7=" + String(growLightOn ? 1 : 0) +
                    "&field8=" + String((sensorFault ? 1 : 0));

  if (!http.begin(client, url))
  {
    Serial.println("ThingSpeak: could not initialize HTTPS request");
    return;
  }
  http.setConnectTimeout(5000);
  http.setTimeout(5000);
  int responseCode = http.GET();
  String entryID = http.getString();

  http.end();

  Serial.print("ThingSpeak HTTP response: ");
  Serial.print(responseCode);
  Serial.print(" | entry id: ");
  Serial.println(entryID);
  if (responseCode < 0)
  {
    Serial.print("ThingSpeak transport error: ");
    Serial.println(HTTPClient::errorToString(responseCode));
  }
}