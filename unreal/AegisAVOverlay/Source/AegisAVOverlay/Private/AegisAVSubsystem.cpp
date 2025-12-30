// Copyright AegisAV Team. All Rights Reserved.

#include "AegisAVSubsystem.h"
#include "AegisAVOverlay.h"
#include "AegisAVWebSocketClient.h"
#include "Widgets/AegisAVMasterHUD.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "Blueprint/UserWidget.h"

void UAegisAVSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    UE_LOG(LogAegisAV, Log, TEXT("AegisAV Subsystem initializing"));

    // Create WebSocket client
    WebSocketClient = NewObject<UAegisAVWebSocketClient>(this);
    WebSocketClient->OnConnectionStateChanged.AddDynamic(this, &UAegisAVSubsystem::OnConnectionStateChanged);

    bOverlayVisible = false;

    // Auto-connect if configured
    if (bAutoConnectOnInit)
    {
        ConnectToBackend(DefaultServerURL);
    }
}

void UAegisAVSubsystem::Deinitialize()
{
    UE_LOG(LogAegisAV, Log, TEXT("AegisAV Subsystem deinitializing"));

    // Cleanup
    DestroyMasterHUD();

    if (WebSocketClient)
    {
        WebSocketClient->Disconnect();
        WebSocketClient = nullptr;
    }

    Super::Deinitialize();
}

void UAegisAVSubsystem::ConnectToBackend(const FString& URL)
{
    if (!WebSocketClient)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("WebSocket client not initialized"));
        return;
    }

    WebSocketClient->Connect(URL);
}

void UAegisAVSubsystem::DisconnectFromBackend()
{
    if (WebSocketClient)
    {
        WebSocketClient->Disconnect();
    }
}

bool UAegisAVSubsystem::IsConnectedToBackend() const
{
    return WebSocketClient && WebSocketClient->IsConnected();
}

void UAegisAVSubsystem::ShowOverlay()
{
    if (!MasterHUD)
    {
        CreateMasterHUD();
    }

    if (MasterHUD)
    {
        MasterHUD->SetVisibility(ESlateVisibility::Visible);
        bOverlayVisible = true;
    }
}

void UAegisAVSubsystem::HideOverlay()
{
    if (MasterHUD)
    {
        MasterHUD->SetVisibility(ESlateVisibility::Collapsed);
        bOverlayVisible = false;
    }
}

void UAegisAVSubsystem::ToggleOverlay()
{
    if (bOverlayVisible)
    {
        HideOverlay();
    }
    else
    {
        ShowOverlay();
    }
}

bool UAegisAVSubsystem::IsOverlayVisible() const
{
    return bOverlayVisible && MasterHUD && MasterHUD->IsVisible();
}

void UAegisAVSubsystem::ShowPanel(FName PanelName)
{
    if (MasterHUD)
    {
        MasterHUD->ShowPanel(PanelName);
    }
}

void UAegisAVSubsystem::HidePanel(FName PanelName)
{
    if (MasterHUD)
    {
        MasterHUD->HidePanel(PanelName);
    }
}

void UAegisAVSubsystem::TogglePanel(FName PanelName)
{
    if (MasterHUD)
    {
        MasterHUD->TogglePanel(PanelName);
    }
}

void UAegisAVSubsystem::CreateMasterHUD()
{
    if (MasterHUD)
    {
        return; // Already created
    }

    // Get the game instance
    UGameInstance* GameInstance = GetGameInstance();
    if (!GameInstance)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("No game instance available"));
        return;
    }

    // Find a player controller to add the widget to
    UWorld* World = GameInstance->GetWorld();
    if (!World)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("No world available"));
        return;
    }

    APlayerController* PC = World->GetFirstPlayerController();
    if (!PC)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("No player controller available"));
        return;
    }

    // Determine widget class to use
    TSubclassOf<UAegisAVMasterHUD> WidgetClass = MasterHUDClass;
    if (!WidgetClass)
    {
        // Use default C++ class if no Blueprint specified
        WidgetClass = UAegisAVMasterHUD::StaticClass();
    }

    // Create the widget
    MasterHUD = CreateWidget<UAegisAVMasterHUD>(PC, WidgetClass);
    if (MasterHUD)
    {
        MasterHUD->Initialize(WebSocketClient);
        MasterHUD->AddToViewport(100); // High Z-order to be on top
        UE_LOG(LogAegisAV, Log, TEXT("AegisAV Master HUD created"));
    }
    else
    {
        UE_LOG(LogAegisAV, Error, TEXT("Failed to create AegisAV Master HUD"));
    }
}

void UAegisAVSubsystem::DestroyMasterHUD()
{
    if (MasterHUD)
    {
        MasterHUD->RemoveFromParent();
        MasterHUD = nullptr;
    }
}

void UAegisAVSubsystem::BindInputActions()
{
    // Input bindings are typically handled in Blueprint or PlayerController
    // This method is a placeholder for future C++ input binding if needed
}

void UAegisAVSubsystem::OnConnectionStateChanged(bool bIsConnected)
{
    if (bIsConnected)
    {
        UE_LOG(LogAegisAV, Log, TEXT("Connected to AegisAV backend"));

        // Auto-show overlay on first connection
        if (bAutoShowOverlay && !bOverlayVisible)
        {
            ShowOverlay();
        }
    }
    else
    {
        UE_LOG(LogAegisAV, Log, TEXT("Disconnected from AegisAV backend"));
    }

    // Forward to MasterHUD for visual indicator
    if (MasterHUD)
    {
        MasterHUD->SetConnectionState(bIsConnected);
    }
}
