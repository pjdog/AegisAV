// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVMasterHUD.h"
#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVOverlay.h"
#include "AegisAVWebSocketClient.h"
#include "Components/CanvasPanel.h"
#include "Components/CanvasPanelSlot.h"
#include "Components/Image.h"
#include "Components/TextBlock.h"
#include "Blueprint/WidgetTree.h"

void UAegisAVMasterHUD::NativeConstruct()
{
    Super::NativeConstruct();

    SetupPanelMap();
    LoadLayout();
}

void UAegisAVMasterHUD::NativeDestruct()
{
    UnbindWebSocketEvents();
    SaveLayout();

    Super::NativeDestruct();
}

void UAegisAVMasterHUD::Initialize(UAegisAVWebSocketClient* InWebSocketClient)
{
    WebSocketClient = InWebSocketClient;

    if (WebSocketClient)
    {
        BindWebSocketEvents();
        bIsConnected = WebSocketClient->IsConnected();
        SetConnectionState(bIsConnected);
    }
}

void UAegisAVMasterHUD::SetupPanelMap()
{
    // Map pre-bound panels
    if (ThoughtBubblePanel)
    {
        PanelMap.Add(FName("ThoughtBubble"), ThoughtBubblePanel);
    }
    if (CameraPanel)
    {
        PanelMap.Add(FName("Camera"), CameraPanel);
    }
    if (CriticsPanel)
    {
        PanelMap.Add(FName("Critics"), CriticsPanel);
    }
    if (TelemetryPanel)
    {
        PanelMap.Add(FName("Telemetry"), TelemetryPanel);
    }
    if (BatteryPanel)
    {
        PanelMap.Add(FName("Battery"), BatteryPanel);
    }

    // Create dynamic panels if not bound
    CreateDynamicPanels();
}

void UAegisAVMasterHUD::CreateDynamicPanels()
{
    // This method creates panels from class references if not already bound in Blueprint
    // For now, panels are expected to be set up in the Blueprint

    // Example of dynamic creation (if panel classes are set):
    /*
    if (!ThoughtBubblePanel && ThoughtBubblePanelClass && MainCanvas)
    {
        ThoughtBubblePanel = CreateWidget<UAegisAVBasePanel>(GetOwningPlayer(), ThoughtBubblePanelClass);
        if (ThoughtBubblePanel)
        {
            MainCanvas->AddChildToCanvas(ThoughtBubblePanel);
            UCanvasPanelSlot* Slot = Cast<UCanvasPanelSlot>(ThoughtBubblePanel->Slot);
            if (Slot)
            {
                Slot->SetPosition(ThoughtBubbleDefaultPos);
                Slot->SetAutoSize(true);
            }
            PanelMap.Add(FName("ThoughtBubble"), ThoughtBubblePanel);
        }
    }
    */
}

void UAegisAVMasterHUD::BindWebSocketEvents()
{
    if (!WebSocketClient)
    {
        return;
    }

    WebSocketClient->OnTelemetryReceived.AddDynamic(this, &UAegisAVMasterHUD::OnTelemetryReceived);
    WebSocketClient->OnAgentThoughtReceived.AddDynamic(this, &UAegisAVMasterHUD::OnAgentThoughtReceived);
    WebSocketClient->OnCriticEvaluationReceived.AddDynamic(this, &UAegisAVMasterHUD::OnCriticEvaluationReceived);
    WebSocketClient->OnBatteryStatusReceived.AddDynamic(this, &UAegisAVMasterHUD::OnBatteryStatusReceived);
    WebSocketClient->OnAnomalyReceived.AddDynamic(this, &UAegisAVMasterHUD::OnAnomalyReceived);
    WebSocketClient->OnCameraFrameReceived.AddDynamic(this, &UAegisAVMasterHUD::OnCameraFrameReceived);
    WebSocketClient->OnConnectionStateChanged.AddDynamic(this, &UAegisAVMasterHUD::OnConnectionStateChanged);
}

void UAegisAVMasterHUD::UnbindWebSocketEvents()
{
    if (!WebSocketClient)
    {
        return;
    }

    WebSocketClient->OnTelemetryReceived.RemoveDynamic(this, &UAegisAVMasterHUD::OnTelemetryReceived);
    WebSocketClient->OnAgentThoughtReceived.RemoveDynamic(this, &UAegisAVMasterHUD::OnAgentThoughtReceived);
    WebSocketClient->OnCriticEvaluationReceived.RemoveDynamic(this, &UAegisAVMasterHUD::OnCriticEvaluationReceived);
    WebSocketClient->OnBatteryStatusReceived.RemoveDynamic(this, &UAegisAVMasterHUD::OnBatteryStatusReceived);
    WebSocketClient->OnAnomalyReceived.RemoveDynamic(this, &UAegisAVMasterHUD::OnAnomalyReceived);
    WebSocketClient->OnCameraFrameReceived.RemoveDynamic(this, &UAegisAVMasterHUD::OnCameraFrameReceived);
    WebSocketClient->OnConnectionStateChanged.RemoveDynamic(this, &UAegisAVMasterHUD::OnConnectionStateChanged);
}

UAegisAVBasePanel* UAegisAVMasterHUD::GetPanelByName(FName PanelName) const
{
    if (const UAegisAVBasePanel* const* Panel = PanelMap.Find(PanelName))
    {
        return const_cast<UAegisAVBasePanel*>(*Panel);
    }
    return nullptr;
}

void UAegisAVMasterHUD::ShowPanel(FName PanelName)
{
    if (UAegisAVBasePanel* Panel = GetPanelByName(PanelName))
    {
        Panel->SetVisibility(ESlateVisibility::Visible);
    }
}

void UAegisAVMasterHUD::HidePanel(FName PanelName)
{
    if (UAegisAVBasePanel* Panel = GetPanelByName(PanelName))
    {
        Panel->SetVisibility(ESlateVisibility::Collapsed);
    }
}

void UAegisAVMasterHUD::TogglePanel(FName PanelName)
{
    if (UAegisAVBasePanel* Panel = GetPanelByName(PanelName))
    {
        if (Panel->IsVisible())
        {
            Panel->SetVisibility(ESlateVisibility::Collapsed);
        }
        else
        {
            Panel->SetVisibility(ESlateVisibility::Visible);
        }
    }
}

bool UAegisAVMasterHUD::IsPanelVisible(FName PanelName) const
{
    if (const UAegisAVBasePanel* Panel = GetPanelByName(PanelName))
    {
        return Panel->IsVisible();
    }
    return false;
}

void UAegisAVMasterHUD::ShowAllPanels()
{
    for (auto& Pair : PanelMap)
    {
        if (Pair.Value)
        {
            Pair.Value->SetVisibility(ESlateVisibility::Visible);
        }
    }
}

void UAegisAVMasterHUD::HideAllPanels()
{
    for (auto& Pair : PanelMap)
    {
        if (Pair.Value)
        {
            Pair.Value->SetVisibility(ESlateVisibility::Collapsed);
        }
    }
}

void UAegisAVMasterHUD::SetConnectionState(bool bConnected)
{
    bIsConnected = bConnected;

    if (ConnectionIndicator)
    {
        // Green when connected, red when disconnected
        FLinearColor Color = bConnected ? FLinearColor::Green : FLinearColor::Red;
        ConnectionIndicator->SetColorAndOpacity(Color);
    }

    if (ConnectionText)
    {
        FText Status = bConnected ? FText::FromString(TEXT("Connected")) : FText::FromString(TEXT("Disconnected"));
        ConnectionText->SetText(Status);
    }
}

void UAegisAVMasterHUD::SaveLayout()
{
    // Save panel positions to config
    // This would typically use GConfig or a save game

    UE_LOG(LogAegisAV, Log, TEXT("Saving AegisAV panel layout"));

    // Example (using game config):
    /*
    for (const auto& Pair : PanelMap)
    {
        if (Pair.Value)
        {
            FString Key = FString::Printf(TEXT("AegisAV.Panel.%s.Position"), *Pair.Key.ToString());
            FVector2D Pos = Pair.Value->GetPanelPosition();
            GConfig->SetString(TEXT("AegisAVOverlay"), *Key, *FString::Printf(TEXT("%f,%f"), Pos.X, Pos.Y), GGameIni);
        }
    }
    GConfig->Flush(false, GGameIni);
    */
}

void UAegisAVMasterHUD::LoadLayout()
{
    UE_LOG(LogAegisAV, Log, TEXT("Loading AegisAV panel layout"));

    // Load panel positions from config or use defaults
    // For now, just apply default positions

    ResetLayout();
}

void UAegisAVMasterHUD::ResetLayout()
{
    // Apply default positions to all panels
    if (UAegisAVBasePanel* Panel = GetPanelByName(FName("ThoughtBubble")))
    {
        Panel->SetPanelPosition(ThoughtBubbleDefaultPos);
    }
    if (UAegisAVBasePanel* Panel = GetPanelByName(FName("Camera")))
    {
        Panel->SetPanelPosition(CameraDefaultPos);
    }
    if (UAegisAVBasePanel* Panel = GetPanelByName(FName("Critics")))
    {
        Panel->SetPanelPosition(CriticsDefaultPos);
    }
    if (UAegisAVBasePanel* Panel = GetPanelByName(FName("Telemetry")))
    {
        Panel->SetPanelPosition(TelemetryDefaultPos);
    }
    if (UAegisAVBasePanel* Panel = GetPanelByName(FName("Battery")))
    {
        Panel->SetPanelPosition(BatteryDefaultPos);
    }
}

// ============================================================================
// WebSocket Event Handlers
// ============================================================================

void UAegisAVMasterHUD::OnTelemetryReceived(const FAegisTelemetry& Telemetry)
{
    // Forward to telemetry panel
    // The panel will have its own update method bound in Blueprint or C++

    // For Blueprint-based panels, you can use a Blueprint event:
    // ReceiveOnTelemetryUpdated(Telemetry);
}

void UAegisAVMasterHUD::OnAgentThoughtReceived(const FAegisAgentThought& Thought)
{
    // Forward to thought bubble panel
}

void UAegisAVMasterHUD::OnCriticEvaluationReceived(const FAegisCriticEvaluation& Evaluation)
{
    // Forward to critics panel
}

void UAegisAVMasterHUD::OnBatteryStatusReceived(const FAegisBatteryStatus& BatteryStatus)
{
    // Forward to battery panel
}

void UAegisAVMasterHUD::OnAnomalyReceived(const FAegisAnomaly& Anomaly)
{
    // Show anomaly alert (could be a temporary popup)
    UE_LOG(LogAegisAV, Log, TEXT("Anomaly detected: %s - %s (Severity: %.2f)"),
           *Anomaly.AnomalyType, *Anomaly.Description, Anomaly.Severity);
}

void UAegisAVMasterHUD::OnCameraFrameReceived(const FString& DroneId, UTexture2D* Frame)
{
    // Forward to camera panel
    // The camera panel will display the texture
}

void UAegisAVMasterHUD::OnConnectionStateChanged(bool bConnected)
{
    SetConnectionState(bConnected);
}
