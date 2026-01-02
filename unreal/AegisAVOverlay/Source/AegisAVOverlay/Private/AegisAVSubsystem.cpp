// Copyright AegisAV Team. All Rights Reserved.

#include "AegisAVSubsystem.h"
#include "AegisAVOverlay.h"
#include "AegisAVWebSocketClient.h"
#include "Widgets/AegisAVMasterHUD.h"
#include "Engine/GameInstance.h"
#include "Engine/World.h"
#include "GameFramework/PlayerController.h"
#include "Blueprint/UserWidget.h"
#include "Misc/ConfigCacheIni.h"
#include "Misc/Paths.h"
#include "Engine/StaticMesh.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Materials/MaterialInterface.h"
#include "Misc/Guid.h"

void UAegisAVSubsystem::Initialize(FSubsystemCollectionBase& Collection)
{
    Super::Initialize(Collection);

    UE_LOG(LogAegisAV, Log, TEXT("AegisAV Subsystem initializing"));

    // Load configuration from AegisAV.ini if it exists
    LoadConfigFromFile();

    // Create WebSocket client
    WebSocketClient = NewObject<UAegisAVWebSocketClient>(this);
    WebSocketClient->OnConnectionStateChanged.AddDynamic(this, &UAegisAVSubsystem::OnConnectionStateChanged);
    BindWebSocketEvents();

    bOverlayVisible = false;

    // Auto-connect if configured
    if (bAutoConnectOnInit)
    {
        ConnectToBackend(DefaultServerURL);
    }
}

void UAegisAVSubsystem::LoadConfigFromFile()
{
    // Try multiple config file locations
    TArray<FString> ConfigPaths = {
        FPaths::ProjectConfigDir() / TEXT("AegisAV.ini"),
        FPaths::ProjectDir() / TEXT("Config/AegisAV.ini"),
        FPaths::EngineConfigDir() / TEXT("AegisAV.ini"),
    };

    FString ConfigPath;
    for (const FString& Path : ConfigPaths)
    {
        if (FPaths::FileExists(Path))
        {
            ConfigPath = Path;
            break;
        }
    }

    if (ConfigPath.IsEmpty())
    {
        UE_LOG(LogAegisAV, Log, TEXT("No AegisAV.ini found, using default settings"));
        return;
    }

    UE_LOG(LogAegisAV, Log, TEXT("Loading config from: %s"), *ConfigPath);

    // Read config values
    FString ServerURL;
    if (GConfig->GetString(TEXT("AegisAV"), TEXT("ServerURL"), ServerURL, ConfigPath))
    {
        DefaultServerURL = ServerURL;
        UE_LOG(LogAegisAV, Log, TEXT("Loaded ServerURL: %s"), *DefaultServerURL);
    }

    bool bAutoConnect = false;
    if (GConfig->GetBool(TEXT("AegisAV"), TEXT("AutoConnect"), bAutoConnect, ConfigPath))
    {
        bAutoConnectOnInit = bAutoConnect;
    }

    bool bAutoShow = true;
    if (GConfig->GetBool(TEXT("AegisAV"), TEXT("AutoShowOverlay"), bAutoShow, ConfigPath))
    {
        bAutoShowOverlay = bAutoShow;
    }

    bool bSpawnAssets = true;
    if (GConfig->GetBool(TEXT("AegisAV"), TEXT("EnableAssetSpawning"), bSpawnAssets, ConfigPath))
    {
        bEnableAssetSpawning = bSpawnAssets;
    }

    bool bUseFirstOrigin = true;
    if (GConfig->GetBool(TEXT("AegisAV"), TEXT("UseFirstAssetAsOrigin"), bUseFirstOrigin, ConfigPath))
    {
        bUseFirstAssetAsOrigin = bUseFirstOrigin;
    }

    FString OriginLatStr;
    const bool bHasOriginLat = GConfig->GetString(TEXT("AegisAV"), TEXT("OriginLatitude"), OriginLatStr, ConfigPath);
    if (bHasOriginLat)
    {
        OriginLatitude = FCString::Atod(*OriginLatStr);
    }

    FString OriginLonStr;
    const bool bHasOriginLon = GConfig->GetString(TEXT("AegisAV"), TEXT("OriginLongitude"), OriginLonStr, ConfigPath);
    if (bHasOriginLon)
    {
        OriginLongitude = FCString::Atod(*OriginLonStr);
    }

    FString OriginAltStr;
    const bool bHasOriginAlt = GConfig->GetString(TEXT("AegisAV"), TEXT("OriginAltitude"), OriginAltStr, ConfigPath);
    if (bHasOriginAlt)
    {
        OriginAltitude = FCString::Atod(*OriginAltStr);
    }

    bOriginInitialized = bHasOriginLat && bHasOriginLon;

    float UnitsScale = UnitsPerMeter;
    if (GConfig->GetFloat(TEXT("AegisAV"), TEXT("UnitsPerMeter"), UnitsScale, ConfigPath))
    {
        UnitsPerMeter = UnitsScale;
    }

    FString MeshPath;
    if (GConfig->GetString(TEXT("AegisAV"), TEXT("AssetMesh.Default"), MeshPath, ConfigPath))
    {
        DefaultAssetMeshPath = MeshPath;
    }
    if (GConfig->GetString(TEXT("AegisAV"), TEXT("AssetMesh.SolarPanel"), MeshPath, ConfigPath))
    {
        SolarPanelMeshPath = MeshPath;
    }
    if (GConfig->GetString(TEXT("AegisAV"), TEXT("AssetMesh.WindTurbine"), MeshPath, ConfigPath))
    {
        WindTurbineMeshPath = MeshPath;
    }
    if (GConfig->GetString(TEXT("AegisAV"), TEXT("AssetMesh.Substation"), MeshPath, ConfigPath))
    {
        SubstationMeshPath = MeshPath;
    }
    if (GConfig->GetString(TEXT("AegisAV"), TEXT("AssetMesh.PowerLine"), MeshPath, ConfigPath))
    {
        PowerLineMeshPath = MeshPath;
    }

    FVector ScaleValue;
    if (GConfig->GetVector(TEXT("AegisAV"), TEXT("AssetScale.Default"), ScaleValue, ConfigPath))
    {
        DefaultAssetScale = ScaleValue;
    }
    if (GConfig->GetVector(TEXT("AegisAV"), TEXT("AssetScale.SolarPanel"), ScaleValue, ConfigPath))
    {
        SolarPanelScale = ScaleValue;
    }
    if (GConfig->GetVector(TEXT("AegisAV"), TEXT("AssetScale.WindTurbine"), ScaleValue, ConfigPath))
    {
        WindTurbineScale = ScaleValue;
    }
    if (GConfig->GetVector(TEXT("AegisAV"), TEXT("AssetScale.Substation"), ScaleValue, ConfigPath))
    {
        SubstationScale = ScaleValue;
    }
    if (GConfig->GetVector(TEXT("AegisAV"), TEXT("AssetScale.PowerLine"), ScaleValue, ConfigPath))
    {
        PowerLineScale = ScaleValue;
    }
}

void UAegisAVSubsystem::Deinitialize()
{
    UE_LOG(LogAegisAV, Log, TEXT("AegisAV Subsystem deinitializing"));

    // Cleanup
    ClearSpawnedActors(SpawnedAssets);
    ClearSpawnedActors(SpawnedAnomalyMarkers);
    DestroyMasterHUD();

    if (WebSocketClient)
    {
        WebSocketClient->OnAssetSpawnReceived.RemoveAll(this);
        WebSocketClient->OnAssetsCleared.RemoveAll(this);
        WebSocketClient->OnAnomalyMarkerReceived.RemoveAll(this);
        WebSocketClient->OnAnomalyMarkersCleared.RemoveAll(this);
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

void UAegisAVSubsystem::BindWebSocketEvents()
{
    if (!WebSocketClient)
    {
        return;
    }

    WebSocketClient->OnAssetSpawnReceived.RemoveAll(this);
    WebSocketClient->OnAssetsCleared.RemoveAll(this);
    WebSocketClient->OnAnomalyMarkerReceived.RemoveAll(this);
    WebSocketClient->OnAnomalyMarkersCleared.RemoveAll(this);

    WebSocketClient->OnAssetSpawnReceived.AddDynamic(this, &UAegisAVSubsystem::OnAssetSpawnReceived);
    WebSocketClient->OnAssetsCleared.AddDynamic(this, &UAegisAVSubsystem::OnAssetsCleared);
    WebSocketClient->OnAnomalyMarkerReceived.AddDynamic(this, &UAegisAVSubsystem::OnAnomalyMarkerReceived);
    WebSocketClient->OnAnomalyMarkersCleared.AddDynamic(this, &UAegisAVSubsystem::OnAnomalyMarkersCleared);
}

void UAegisAVSubsystem::OnAssetSpawnReceived(const FAegisAssetSpawn& Asset)
{
    if (!bEnableAssetSpawning)
    {
        return;
    }

    InitializeOriginIfNeeded(Asset);
    if (!bOriginInitialized)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Asset spawn ignored: origin not initialized"));
        return;
    }

    const FVector WorldLocation = GeoToWorld(Asset.Latitude, Asset.Longitude, Asset.AltitudeM);
    const FRotator Rotation(0.0f, Asset.RotationDeg, 0.0f);
    const float ScaleMultiplier = Asset.Scale > 0.0f ? Asset.Scale : 1.0f;
    const FVector AssetScale = GetScaleForAssetType(Asset.AssetType) * ScaleMultiplier;

    FString AssetKey = Asset.AssetId;
    if (AssetKey.IsEmpty())
    {
        AssetKey = Asset.Name;
    }
    if (AssetKey.IsEmpty())
    {
        AssetKey = FString::Printf(TEXT("asset_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
    }

    AActor* AssetActor = nullptr;
    if (TWeakObjectPtr<AActor>* Existing = SpawnedAssets.Find(AssetKey))
    {
        AssetActor = Existing->Get();
    }

    if (!AssetActor)
    {
        AssetActor = SpawnMeshActor(AssetKey, Asset.AssetType, Asset.Name);
        if (!AssetActor)
        {
            return;
        }
        SpawnedAssets.Add(AssetKey, AssetActor);
    }

    AssetActor->SetActorLocation(WorldLocation);
    AssetActor->SetActorRotation(Rotation);
    AssetActor->SetActorScale3D(AssetScale);
}

void UAegisAVSubsystem::OnAssetsCleared()
{
    ClearSpawnedActors(SpawnedAssets);
}

void UAegisAVSubsystem::OnAnomalyMarkerReceived(const FAegisAnomalyMarker& Marker)
{
    if (!bEnableAssetSpawning)
    {
        return;
    }

    if (!bOriginInitialized && bUseFirstAssetAsOrigin)
    {
        OriginLatitude = Marker.Latitude;
        OriginLongitude = Marker.Longitude;
        OriginAltitude = Marker.AltitudeM;
        bOriginInitialized = true;
        UE_LOG(LogAegisAV, Log, TEXT("Initialized geo origin from anomaly marker (%0.6f, %0.6f, %0.2f)"),
            OriginLatitude, OriginLongitude, OriginAltitude);
    }

    if (!bOriginInitialized)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Anomaly marker ignored: origin not initialized"));
        return;
    }

    const FVector WorldLocation = GeoToWorld(Marker.Latitude, Marker.Longitude, Marker.AltitudeM);

    FString MarkerKey = Marker.AnomalyId;
    if (MarkerKey.IsEmpty())
    {
        MarkerKey = Marker.AssetId;
    }
    if (MarkerKey.IsEmpty())
    {
        MarkerKey = FString::Printf(TEXT("marker_%s"), *FGuid::NewGuid().ToString(EGuidFormats::Digits));
    }

    AActor* MarkerActor = nullptr;
    if (TWeakObjectPtr<AActor>* Existing = SpawnedAnomalyMarkers.Find(MarkerKey))
    {
        MarkerActor = Existing->Get();
    }

    if (!MarkerActor)
    {
        MarkerActor = SpawnMeshActor(MarkerKey, Marker.MarkerType.IsEmpty() ? TEXT("anomaly_marker") : Marker.MarkerType, Marker.Label);
        if (!MarkerActor)
        {
            return;
        }
        SpawnedAnomalyMarkers.Add(MarkerKey, MarkerActor);

        if (AStaticMeshActor* MeshActor = Cast<AStaticMeshActor>(MarkerActor))
        {
            if (UStaticMeshComponent* MeshComp = MeshActor->GetStaticMeshComponent())
            {
                MeshComp->SetCollisionEnabled(ECollisionEnabled::NoCollision);
                if (UMaterialInterface* BaseMat = MeshComp->GetMaterial(0))
                {
                    if (UMaterialInstanceDynamic* MID = UMaterialInstanceDynamic::Create(BaseMat, this))
                    {
                        MID->SetVectorParameterValue(TEXT("Color"), Marker.Color);
                        MID->SetVectorParameterValue(TEXT("BaseColor"), Marker.Color);
                        MeshComp->SetMaterial(0, MID);
                    }
                }
            }
        }
    }

    const float VisualScale = FMath::Clamp(0.5f + Marker.Severity * 0.5f, 0.35f, 4.0f);
    MarkerActor->SetActorLocation(WorldLocation);
    MarkerActor->SetActorScale3D(FVector(VisualScale));
}

void UAegisAVSubsystem::OnAnomalyMarkersCleared()
{
    ClearSpawnedActors(SpawnedAnomalyMarkers);
}

FVector UAegisAVSubsystem::GeoToWorld(double Latitude, double Longitude, double AltitudeM) const
{
    if (!bOriginInitialized)
    {
        return FVector::ZeroVector;
    }

    static const double EarthRadiusM = 6371000.0;
    static const double DegToRad = PI / 180.0;

    const double DeltaLat = (Latitude - OriginLatitude) * DegToRad;
    const double DeltaLon = (Longitude - OriginLongitude) * DegToRad;
    const double MeanLat = (Latitude + OriginLatitude) * 0.5 * DegToRad;

    const double NorthM = DeltaLat * EarthRadiusM;
    const double EastM = DeltaLon * EarthRadiusM * FMath::Cos(static_cast<float>(MeanLat));
    const double UpM = AltitudeM - OriginAltitude;

    return FVector(
        static_cast<float>(NorthM * UnitsPerMeter),
        static_cast<float>(EastM * UnitsPerMeter),
        static_cast<float>(UpM * UnitsPerMeter));
}

UStaticMesh* UAegisAVSubsystem::LoadMeshForAssetType(const FString& AssetType) const
{
    FString Normalized = AssetType;
    Normalized.ToLowerInline();

    FString MeshPath = DefaultAssetMeshPath;
    if (Normalized.Contains(TEXT("solar")))
    {
        MeshPath = SolarPanelMeshPath;
    }
    else if (Normalized.Contains(TEXT("wind")))
    {
        MeshPath = WindTurbineMeshPath;
    }
    else if (Normalized.Contains(TEXT("substation")) || Normalized.Contains(TEXT("transformer")))
    {
        MeshPath = SubstationMeshPath;
    }
    else if (Normalized.Contains(TEXT("powerline")) || Normalized.Contains(TEXT("power_line")) || Normalized.Contains(TEXT("line")))
    {
        MeshPath = PowerLineMeshPath;
    }
    else if (Normalized.Contains(TEXT("anomaly")) || Normalized.Contains(TEXT("marker")))
    {
        MeshPath = TEXT("/Engine/BasicShapes/Sphere.Sphere");
    }

    UStaticMesh* Mesh = Cast<UStaticMesh>(StaticLoadObject(UStaticMesh::StaticClass(), nullptr, *MeshPath));
    if (!Mesh)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Failed to load mesh '%s' for asset type '%s'"), *MeshPath, *AssetType);
    }

    return Mesh;
}

FVector UAegisAVSubsystem::GetScaleForAssetType(const FString& AssetType) const
{
    FString Normalized = AssetType;
    Normalized.ToLowerInline();

    if (Normalized.Contains(TEXT("solar")))
    {
        return SolarPanelScale;
    }
    if (Normalized.Contains(TEXT("wind")))
    {
        return WindTurbineScale;
    }
    if (Normalized.Contains(TEXT("substation")) || Normalized.Contains(TEXT("transformer")))
    {
        return SubstationScale;
    }
    if (Normalized.Contains(TEXT("powerline")) || Normalized.Contains(TEXT("power_line")) || Normalized.Contains(TEXT("line")))
    {
        return PowerLineScale;
    }

    return DefaultAssetScale;
}

void UAegisAVSubsystem::InitializeOriginIfNeeded(const FAegisAssetSpawn& Asset)
{
    if (bOriginInitialized || !bUseFirstAssetAsOrigin)
    {
        return;
    }

    OriginLatitude = Asset.Latitude;
    OriginLongitude = Asset.Longitude;
    OriginAltitude = Asset.AltitudeM;
    bOriginInitialized = true;

    UE_LOG(LogAegisAV, Log, TEXT("Initialized geo origin from asset %s (%0.6f, %0.6f, %0.2f)"),
        *Asset.AssetId, OriginLatitude, OriginLongitude, OriginAltitude);
}

AActor* UAegisAVSubsystem::SpawnMeshActor(const FString& AssetId, const FString& AssetType, const FString& Name)
{
    UWorld* World = GetWorld();
    if (!World)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Cannot spawn asset '%s': no world available"), *AssetId);
        return nullptr;
    }

    UStaticMesh* Mesh = LoadMeshForAssetType(AssetType);
    if (!Mesh)
    {
        return nullptr;
    }

    FActorSpawnParameters SpawnParams;
    SpawnParams.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
    AStaticMeshActor* MeshActor = World->SpawnActor<AStaticMeshActor>(AStaticMeshActor::StaticClass(), FVector::ZeroVector, FRotator::ZeroRotator, SpawnParams);
    if (!MeshActor)
    {
        UE_LOG(LogAegisAV, Warning, TEXT("Failed to spawn asset actor for '%s'"), *AssetId);
        return nullptr;
    }

    if (UStaticMeshComponent* MeshComp = MeshActor->GetStaticMeshComponent())
    {
        MeshComp->SetStaticMesh(Mesh);
        MeshComp->SetMobility(EComponentMobility::Movable);
    }

    MeshActor->Tags.Add(TEXT("AegisAsset"));

#if WITH_EDITOR
    if (!Name.IsEmpty())
    {
        MeshActor->SetActorLabel(Name);
    }
#endif

    return MeshActor;
}

void UAegisAVSubsystem::ClearSpawnedActors(TMap<FString, TWeakObjectPtr<AActor>>& ActorMap)
{
    for (TPair<FString, TWeakObjectPtr<AActor>>& Pair : ActorMap)
    {
        if (AActor* Actor = Pair.Value.Get())
        {
            Actor->Destroy();
        }
    }

    ActorMap.Empty();
}
