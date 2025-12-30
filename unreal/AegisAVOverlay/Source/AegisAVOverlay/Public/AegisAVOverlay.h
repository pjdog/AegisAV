// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

DECLARE_LOG_CATEGORY_EXTERN(LogAegisAV, Log, All);

/**
 * AegisAV Overlay Module
 *
 * Provides native UMG overlay widgets for the AegisAV drone simulation system.
 * Connects to the AegisAV Python backend via WebSocket to receive real-time
 * telemetry, agent thinking states, critic evaluations, and camera frames.
 */
class FAegisAVOverlayModule : public IModuleInterface
{
public:
    /**
     * IModuleInterface implementation
     */
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    /**
     * Get the module instance
     */
    static inline FAegisAVOverlayModule& Get()
    {
        return FModuleManager::LoadModuleChecked<FAegisAVOverlayModule>("AegisAVOverlay");
    }

    /**
     * Check if the module is loaded
     */
    static inline bool IsAvailable()
    {
        return FModuleManager::Get().IsModuleLoaded("AegisAVOverlay");
    }
};
