// Copyright AegisAV Team. All Rights Reserved.

using UnrealBuildTool;

public class AegisAVOverlay : ModuleRules
{
    public AegisAVOverlay(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        // Public dependencies - available to other modules
        PublicDependencyModuleNames.AddRange(new string[]
        {
            "Core",
            "CoreUObject",
            "Engine",
            "InputCore",
            "UMG",
            "Slate",
            "SlateCore"
        });

        // Private dependencies - only used internally
        PrivateDependencyModuleNames.AddRange(new string[]
        {
            "WebSockets",           // Built-in UE WebSocket module
            "Json",                 // JSON parsing
            "JsonUtilities",        // JSON to UStruct conversion
            "ImageWrapper",         // PNG decoding for camera frames
            "RenderCore",           // Dynamic texture creation
            "RHI"                   // Low-level rendering
        });

        // Include paths
        PublicIncludePaths.AddRange(new string[]
        {
            // Add public include paths here
        });

        PrivateIncludePaths.AddRange(new string[]
        {
            // Add private include paths here
        });

        // Uncomment if using online features
        // PrivateDependencyModuleNames.Add("OnlineSubsystem");

        // Platform-specific settings
        if (Target.Platform == UnrealTargetPlatform.Win64)
        {
            // Windows-specific dependencies if needed
        }
    }
}
