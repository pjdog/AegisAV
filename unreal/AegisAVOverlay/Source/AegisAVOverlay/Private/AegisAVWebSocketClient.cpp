// Copyright AegisAV Team. All Rights Reserved.

#include "AegisAVWebSocketClient.h"
#include "AegisAVOverlay.h"
#include "WebSocketsModule.h"
#include "IWebSocket.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "Misc/Base64.h"
#include "IImageWrapper.h"
#include "IImageWrapperModule.h"
#include "Engine/Texture2D.h"
#include "TimerManager.h"
#include "Engine/World.h"
#include "Engine/Engine.h"

UAegisAVWebSocketClient::UAegisAVWebSocketClient()
    : ServerURL(TEXT("ws://localhost:8080/ws/unreal"))
    , bAutoReconnect(true)
    , ReconnectInterval(5.0f)
    , bIsConnected(false)
    , ReconnectAttempts(0)
{
}

UAegisAVWebSocketClient::~UAegisAVWebSocketClient()
{
    Disconnect();
}

void UAegisAVWebSocketClient::Connect(const FString& URL)
{
    // Disconnect existing connection
    if (WebSocket.IsValid())
    {
        Disconnect();
    }

    ServerURL = URL;
    ReconnectAttempts = 0;

    // Ensure WebSockets module is loaded
    if (!FModuleManager::Get().IsModuleLoaded("WebSockets"))
    {
        FModuleManager::Get().LoadModule("WebSockets");
    }

    UE_LOG(LogAegisAV, Log, TEXT("Connecting to AegisAV server: %s"), *ServerURL);

    // Create WebSocket
    WebSocket = FWebSocketsModule::Get().CreateWebSocket(ServerURL, TEXT("ws"));

    // Bind callbacks
    WebSocket->OnConnected().AddUObject(this, &UAegisAVWebSocketClient::OnWebSocketConnected);
    WebSocket->OnConnectionError().AddUObject(this, &UAegisAVWebSocketClient::OnWebSocketConnectionError);
    WebSocket->OnClosed().AddUObject(this, &UAegisAVWebSocketClient::OnWebSocketClosed);
    WebSocket->OnMessage().AddUObject(this, &UAegisAVWebSocketClient::OnWebSocketMessage);
    WebSocket->OnRawMessage().AddUObject(this, &UAegisAVWebSocketClient::OnWebSocketRawMessage);

    // Connect
    WebSocket->Connect();
}

void UAegisAVWebSocketClient::Disconnect()
{
    StopReconnectTimer();

    if (WebSocket.IsValid())
    {
        WebSocket->Close();
        WebSocket.Reset();
    }

    if (bIsConnected)
    {
        bIsConnected = false;
        OnConnectionStateChanged.Broadcast(false);
    }
}

bool UAegisAVWebSocketClient::IsConnected() const
{
    return bIsConnected && WebSocket.IsValid() && WebSocket->IsConnected();
}

void UAegisAVWebSocketClient::SetAutoReconnect(bool bEnable, float IntervalSeconds)
{
    bAutoReconnect = bEnable;
    ReconnectInterval = FMath::Max(1.0f, IntervalSeconds);

    if (!bEnable)
    {
        StopReconnectTimer();
    }
}

void UAegisAVWebSocketClient::OnWebSocketConnected()
{
    UE_LOG(LogAegisAV, Log, TEXT("Connected to AegisAV server"));

    bIsConnected = true;
    ReconnectAttempts = 0;
    StopReconnectTimer();

    OnConnectionStateChanged.Broadcast(true);
}

void UAegisAVWebSocketClient::OnWebSocketConnectionError(const FString& Error)
{
    UE_LOG(LogAegisAV, Warning, TEXT("WebSocket connection error: %s"), *Error);

    bIsConnected = false;
    OnConnectionStateChanged.Broadcast(false);

    if (bAutoReconnect)
    {
        AttemptReconnect();
    }
}

void UAegisAVWebSocketClient::OnWebSocketClosed(int32 StatusCode, const FString& Reason, bool bWasClean)
{
    UE_LOG(LogAegisAV, Log, TEXT("WebSocket closed (Code: %d, Reason: %s, Clean: %s)"),
           StatusCode, *Reason, bWasClean ? TEXT("Yes") : TEXT("No"));

    bIsConnected = false;
    OnConnectionStateChanged.Broadcast(false);

    if (bAutoReconnect && !bWasClean)
    {
        AttemptReconnect();
    }
}

void UAegisAVWebSocketClient::OnWebSocketMessage(const FString& Message)
{
    ParseAndDispatchMessage(Message);
}

void UAegisAVWebSocketClient::OnWebSocketRawMessage(const void* Data, SIZE_T Size, SIZE_T BytesRemaining)
{
    // Handle binary messages if needed (camera frames could be sent as binary)
    // For now, we use base64-encoded JSON
}

void UAegisAVWebSocketClient::ParseAndDispatchMessage(const FString& JsonString)
{
    TSharedPtr<FJsonObject> JsonObject;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonString);

    if (!FJsonSerializer::Deserialize(Reader, JsonObject) || !JsonObject.IsValid())
    {
        UE_LOG(LogAegisAV, Verbose, TEXT("Failed to parse JSON message"));
        return;
    }

    FString MessageType;
    if (!JsonObject->TryGetStringField(TEXT("type"), MessageType))
    {
        UE_LOG(LogAegisAV, Verbose, TEXT("Message missing 'type' field"));
        return;
    }

    // Dispatch based on message type
    if (MessageType == TEXT("telemetry"))
    {
        ParseTelemetryMessage(JsonObject);
    }
    else if (MessageType == TEXT("thinking_update") || MessageType == TEXT("agent_thinking"))
    {
        ParseAgentThinkingMessage(JsonObject);
    }
    else if (MessageType == TEXT("critic_result"))
    {
        ParseCriticResultMessage(JsonObject);
    }
    else if (MessageType == TEXT("battery_update"))
    {
        ParseBatteryUpdateMessage(JsonObject);
    }
    else if (MessageType == TEXT("anomaly_detected"))
    {
        ParseAnomalyMessage(JsonObject);
    }
    else if (MessageType == TEXT("camera_frame"))
    {
        ParseCameraFrameMessage(JsonObject);
    }
    // Add more message types as needed
}

void UAegisAVWebSocketClient::ParseTelemetryMessage(const TSharedPtr<FJsonObject>& JsonObject)
{
    FAegisTelemetry Telemetry;

    JsonObject->TryGetStringField(TEXT("drone_id"), Telemetry.DroneId);
    JsonObject->TryGetNumberField(TEXT("timestamp_ms"), Telemetry.Timestamp);
    JsonObject->TryGetNumberField(TEXT("battery_percent"), Telemetry.BatteryPercent);
    JsonObject->TryGetNumberField(TEXT("altitude_m"), Telemetry.AltitudeM);
    JsonObject->TryGetNumberField(TEXT("speed_ms"), Telemetry.SpeedMs);

    // Parse position
    const TSharedPtr<FJsonObject>* PosObj;
    if (JsonObject->TryGetObjectField(TEXT("position"), PosObj))
    {
        (*PosObj)->TryGetNumberField(TEXT("x"), Telemetry.Position.X);
        (*PosObj)->TryGetNumberField(TEXT("y"), Telemetry.Position.Y);
        (*PosObj)->TryGetNumberField(TEXT("z"), Telemetry.Position.Z);
    }

    // Parse velocity
    const TSharedPtr<FJsonObject>* VelObj;
    if (JsonObject->TryGetObjectField(TEXT("velocity"), VelObj))
    {
        (*VelObj)->TryGetNumberField(TEXT("x"), Telemetry.Velocity.X);
        (*VelObj)->TryGetNumberField(TEXT("y"), Telemetry.Velocity.Y);
        (*VelObj)->TryGetNumberField(TEXT("z"), Telemetry.Velocity.Z);
    }

    // Parse attitude
    const TSharedPtr<FJsonObject>* AttObj;
    if (JsonObject->TryGetObjectField(TEXT("attitude"), AttObj))
    {
        double Pitch, Yaw, Roll;
        if ((*AttObj)->TryGetNumberField(TEXT("pitch"), Pitch))
            Telemetry.Attitude.Pitch = Pitch;
        if ((*AttObj)->TryGetNumberField(TEXT("yaw"), Yaw))
            Telemetry.Attitude.Yaw = Yaw;
        if ((*AttObj)->TryGetNumberField(TEXT("roll"), Roll))
            Telemetry.Attitude.Roll = Roll;
    }

    OnTelemetryReceived.Broadcast(Telemetry);
}

void UAegisAVWebSocketClient::ParseAgentThinkingMessage(const TSharedPtr<FJsonObject>& JsonObject)
{
    FAegisAgentThought Thought;

    JsonObject->TryGetStringField(TEXT("drone_id"), Thought.DroneId);
    JsonObject->TryGetStringField(TEXT("cognitive_level"), Thought.CognitiveLevel);
    JsonObject->TryGetStringField(TEXT("urgency"), Thought.Urgency);
    JsonObject->TryGetStringField(TEXT("current_goal"), Thought.CurrentGoal);
    JsonObject->TryGetStringField(TEXT("target_asset"), Thought.TargetAsset);
    JsonObject->TryGetStringField(TEXT("situation"), Thought.Situation);
    JsonObject->TryGetStringField(TEXT("decision_action"), Thought.SelectedDecision);
    JsonObject->TryGetNumberField(TEXT("decision_confidence"), Thought.Confidence);
    JsonObject->TryGetNumberField(TEXT("risk_score"), Thought.RiskScore);
    JsonObject->TryGetNumberField(TEXT("timestamp_ms"), Thought.Timestamp);

    // Parse risk level
    FString RiskLevelStr;
    if (JsonObject->TryGetStringField(TEXT("risk_level"), RiskLevelStr))
    {
        Thought.RiskLevel = StringToRiskLevel(RiskLevelStr);
    }

    // Parse considerations array
    const TArray<TSharedPtr<FJsonValue>>* ConsiderationsArray;
    if (JsonObject->TryGetArrayField(TEXT("considerations"), ConsiderationsArray))
    {
        for (const auto& Item : *ConsiderationsArray)
        {
            FString Value;
            if (Item->TryGetString(Value))
            {
                Thought.Considerations.Add(Value);
            }
        }
    }

    // Parse options array
    const TArray<TSharedPtr<FJsonValue>>* OptionsArray;
    if (JsonObject->TryGetArrayField(TEXT("options"), OptionsArray))
    {
        for (const auto& Item : *OptionsArray)
        {
            FString Value;
            if (Item->TryGetString(Value))
            {
                Thought.Options.Add(Value);
            }
        }
    }

    OnAgentThoughtReceived.Broadcast(Thought);
}

void UAegisAVWebSocketClient::ParseCriticResultMessage(const TSharedPtr<FJsonObject>& JsonObject)
{
    FAegisCriticEvaluation Evaluation;

    JsonObject->TryGetStringField(TEXT("drone_id"), Evaluation.DroneId);
    JsonObject->TryGetNumberField(TEXT("timestamp_ms"), Evaluation.Timestamp);

    // Parse critics object
    const TSharedPtr<FJsonObject>* CriticsObj;
    if (JsonObject->TryGetObjectField(TEXT("critics"), CriticsObj))
    {
        // Safety critic
        const TSharedPtr<FJsonObject>* SafetyObj;
        if ((*CriticsObj)->TryGetObjectField(TEXT("safety"), SafetyObj))
        {
            FString Verdict;
            if ((*SafetyObj)->TryGetStringField(TEXT("verdict"), Verdict))
            {
                Evaluation.SafetyCritic.Verdict = StringToVerdict(Verdict);
            }
            (*SafetyObj)->TryGetNumberField(TEXT("confidence"), Evaluation.SafetyCritic.Confidence);
            Evaluation.SafetyCritic.CriticName = TEXT("Safety");
        }

        // Efficiency critic
        const TSharedPtr<FJsonObject>* EfficiencyObj;
        if ((*CriticsObj)->TryGetObjectField(TEXT("efficiency"), EfficiencyObj))
        {
            FString Verdict;
            if ((*EfficiencyObj)->TryGetStringField(TEXT("verdict"), Verdict))
            {
                Evaluation.EfficiencyCritic.Verdict = StringToVerdict(Verdict);
            }
            (*EfficiencyObj)->TryGetNumberField(TEXT("confidence"), Evaluation.EfficiencyCritic.Confidence);
            Evaluation.EfficiencyCritic.CriticName = TEXT("Efficiency");
        }

        // Goal alignment critic
        const TSharedPtr<FJsonObject>* GoalObj;
        if ((*CriticsObj)->TryGetObjectField(TEXT("goal_alignment"), GoalObj))
        {
            FString Verdict;
            if ((*GoalObj)->TryGetStringField(TEXT("verdict"), Verdict))
            {
                Evaluation.GoalAlignmentCritic.Verdict = StringToVerdict(Verdict);
            }
            (*GoalObj)->TryGetNumberField(TEXT("confidence"), Evaluation.GoalAlignmentCritic.Confidence);
            Evaluation.GoalAlignmentCritic.CriticName = TEXT("Goal Alignment");
        }
    }

    OnCriticEvaluationReceived.Broadcast(Evaluation);
}

void UAegisAVWebSocketClient::ParseBatteryUpdateMessage(const TSharedPtr<FJsonObject>& JsonObject)
{
    FAegisBatteryStatus Battery;

    JsonObject->TryGetStringField(TEXT("drone_id"), Battery.DroneId);
    JsonObject->TryGetNumberField(TEXT("percent"), Battery.Percent);
    JsonObject->TryGetNumberField(TEXT("voltage"), Battery.Voltage);
    JsonObject->TryGetNumberField(TEXT("current"), Battery.Current);
    JsonObject->TryGetNumberField(TEXT("temperature_c"), Battery.TemperatureC);
    JsonObject->TryGetNumberField(TEXT("time_remaining_s"), Battery.TimeRemainingS);
    JsonObject->TryGetNumberField(TEXT("distance_remaining_m"), Battery.DistanceRemainingM);
    JsonObject->TryGetBoolField(TEXT("is_charging"), Battery.bIsCharging);
    JsonObject->TryGetBoolField(TEXT("is_critical"), Battery.bIsCritical);
    JsonObject->TryGetBoolField(TEXT("is_low"), Battery.bIsLow);
    JsonObject->TryGetNumberField(TEXT("timestamp_ms"), Battery.Timestamp);

    OnBatteryStatusReceived.Broadcast(Battery);
}

void UAegisAVWebSocketClient::ParseAnomalyMessage(const TSharedPtr<FJsonObject>& JsonObject)
{
    FAegisAnomaly Anomaly;

    JsonObject->TryGetStringField(TEXT("anomaly_id"), Anomaly.AnomalyId);
    JsonObject->TryGetStringField(TEXT("drone_id"), Anomaly.DroneId);
    JsonObject->TryGetStringField(TEXT("anomaly_type"), Anomaly.AnomalyType);
    JsonObject->TryGetStringField(TEXT("description"), Anomaly.Description);
    JsonObject->TryGetNumberField(TEXT("severity"), Anomaly.Severity);
    JsonObject->TryGetNumberField(TEXT("timestamp_ms"), Anomaly.Timestamp);

    // Parse location
    double Lat, Lon, Alt;
    if (JsonObject->TryGetNumberField(TEXT("latitude"), Lat) &&
        JsonObject->TryGetNumberField(TEXT("longitude"), Lon))
    {
        JsonObject->TryGetNumberField(TEXT("altitude_m"), Alt);
        Anomaly.Location = FVector(Lat, Lon, Alt);
    }

    OnAnomalyReceived.Broadcast(Anomaly);
}

void UAegisAVWebSocketClient::ParseCameraFrameMessage(const TSharedPtr<FJsonObject>& JsonObject)
{
    FString DroneId;
    FString Base64Image;
    int32 Width = 1280;
    int32 Height = 720;

    JsonObject->TryGetStringField(TEXT("drone_id"), DroneId);
    JsonObject->TryGetStringField(TEXT("image_base64"), Base64Image);
    JsonObject->TryGetNumberField(TEXT("width"), Width);
    JsonObject->TryGetNumberField(TEXT("height"), Height);

    if (Base64Image.IsEmpty())
    {
        return;
    }

    // Decode base64 to texture
    UTexture2D* Texture = Base64ToTexture2D(Base64Image, Width, Height);
    if (Texture)
    {
        OnCameraFrameReceived.Broadcast(DroneId, Texture);
    }
}

UTexture2D* UAegisAVWebSocketClient::Base64ToTexture2D(const FString& Base64String, int32 Width, int32 Height)
{
    // Decode base64
    TArray<uint8> DecodedBytes;
    if (!FBase64::Decode(Base64String, DecodedBytes))
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Failed to decode base64 image data"));
        return nullptr;
    }

    // Load image wrapper module
    IImageWrapperModule& ImageWrapperModule = FModuleManager::LoadModuleChecked<IImageWrapperModule>("ImageWrapper");
    TSharedPtr<IImageWrapper> ImageWrapper = ImageWrapperModule.CreateImageWrapper(EImageFormat::PNG);

    // Set compressed data
    if (!ImageWrapper->SetCompressed(DecodedBytes.GetData(), DecodedBytes.Num()))
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Failed to set compressed image data"));
        return nullptr;
    }

    // Get raw BGRA data
    TArray<uint8> RawData;
    if (!ImageWrapper->GetRaw(ERGBFormat::BGRA, 8, RawData))
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Failed to get raw image data"));
        return nullptr;
    }

    // Get actual dimensions from image
    Width = ImageWrapper->GetWidth();
    Height = ImageWrapper->GetHeight();

    // Create transient texture
    UTexture2D* Texture = UTexture2D::CreateTransient(Width, Height, PF_B8G8R8A8);
    if (!Texture)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Failed to create transient texture"));
        return nullptr;
    }

    // Copy data to texture
    void* TextureData = Texture->GetPlatformData()->Mips[0].BulkData.Lock(LOCK_READ_WRITE);
    FMemory::Memcpy(TextureData, RawData.GetData(), RawData.Num());
    Texture->GetPlatformData()->Mips[0].BulkData.Unlock();

    // Update resource
    Texture->UpdateResource();

    return Texture;
}

ECriticVerdict UAegisAVWebSocketClient::StringToVerdict(const FString& VerdictString) const
{
    if (VerdictString.Equals(TEXT("approve"), ESearchCase::IgnoreCase))
    {
        return ECriticVerdict::Approve;
    }
    else if (VerdictString.Equals(TEXT("approve_with_concerns"), ESearchCase::IgnoreCase))
    {
        return ECriticVerdict::ApproveWithConcerns;
    }
    else if (VerdictString.Equals(TEXT("escalate"), ESearchCase::IgnoreCase))
    {
        return ECriticVerdict::Escalate;
    }
    else if (VerdictString.Equals(TEXT("reject"), ESearchCase::IgnoreCase))
    {
        return ECriticVerdict::Reject;
    }
    return ECriticVerdict::Approve;
}

ERiskLevel UAegisAVWebSocketClient::StringToRiskLevel(const FString& RiskString) const
{
    if (RiskString.Equals(TEXT("low"), ESearchCase::IgnoreCase))
    {
        return ERiskLevel::Low;
    }
    else if (RiskString.Equals(TEXT("medium"), ESearchCase::IgnoreCase))
    {
        return ERiskLevel::Medium;
    }
    else if (RiskString.Equals(TEXT("high"), ESearchCase::IgnoreCase))
    {
        return ERiskLevel::High;
    }
    else if (RiskString.Equals(TEXT("critical"), ESearchCase::IgnoreCase))
    {
        return ERiskLevel::Critical;
    }
    return ERiskLevel::Low;
}

void UAegisAVWebSocketClient::AttemptReconnect()
{
    if (!bAutoReconnect)
    {
        return;
    }

    ReconnectAttempts++;
    float Delay = FMath::Min(ReconnectInterval * ReconnectAttempts, 30.0f); // Cap at 30 seconds

    UE_LOG(LogAegisAV, Log, TEXT("Attempting reconnection in %.1f seconds (attempt %d)"), Delay, ReconnectAttempts);

    // Use world timer if available, otherwise use a simple delay
    if (GEngine && GEngine->GetWorldContexts().Num() > 0)
    {
        UWorld* World = GEngine->GetWorldContexts()[0].World();
        if (World)
        {
            World->GetTimerManager().SetTimer(
                ReconnectTimerHandle,
                [this]()
                {
                    if (!IsConnected())
                    {
                        Connect(ServerURL);
                    }
                },
                Delay,
                false
            );
        }
    }
}

void UAegisAVWebSocketClient::StopReconnectTimer()
{
    if (ReconnectTimerHandle.IsValid())
    {
        if (GEngine && GEngine->GetWorldContexts().Num() > 0)
        {
            UWorld* World = GEngine->GetWorldContexts()[0].World();
            if (World)
            {
                World->GetTimerManager().ClearTimer(ReconnectTimerHandle);
            }
        }
        ReconnectTimerHandle.Invalidate();
    }
}
