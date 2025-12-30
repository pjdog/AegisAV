// Copyright AegisAV Team. All Rights Reserved.

#include "AegisAVOverlay.h"
#include "Modules/ModuleManager.h"

DEFINE_LOG_CATEGORY(LogAegisAV);

#define LOCTEXT_NAMESPACE "FAegisAVOverlayModule"

void FAegisAVOverlayModule::StartupModule()
{
    UE_LOG(LogAegisAV, Log, TEXT("AegisAV Overlay module starting up"));

    // Module initialization code
    // The actual WebSocket connection and UI creation is handled by UAegisAVSubsystem
}

void FAegisAVOverlayModule::ShutdownModule()
{
    UE_LOG(LogAegisAV, Log, TEXT("AegisAV Overlay module shutting down"));

    // Cleanup code if needed
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FAegisAVOverlayModule, AegisAVOverlay)
