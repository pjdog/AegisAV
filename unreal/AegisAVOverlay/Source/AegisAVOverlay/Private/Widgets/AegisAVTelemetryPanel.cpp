// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVTelemetryPanel.h"
#include "Components/TextBlock.h"

UAegisAVTelemetryPanel::UAegisAVTelemetryPanel(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    PanelTitle = FText::FromString(TEXT("Telemetry"));
    ExpandedSize = FVector2D(280.0f, 220.0f);
}

void UAegisAVTelemetryPanel::NativeConstruct()
{
    Super::NativeConstruct();

    // Initialize with zero values
    CurrentTelemetry = FAegisTelemetry();
    UpdateTelemetry(CurrentTelemetry);
}

void UAegisAVTelemetryPanel::UpdateTelemetry(const FAegisTelemetry& Telemetry)
{
    CurrentTelemetry = Telemetry;

    // Update drone ID
    if (DroneIdText)
    {
        DroneIdText->SetText(FText::FromString(Telemetry.DroneId.IsEmpty() ? TEXT("---") : Telemetry.DroneId));
    }

    // Update position
    if (PositionXText)
    {
        PositionXText->SetText(FText::FromString(FString::Printf(TEXT("%.2f"), Telemetry.Position.X)));
    }
    if (PositionYText)
    {
        PositionYText->SetText(FText::FromString(FString::Printf(TEXT("%.2f"), Telemetry.Position.Y)));
    }
    if (PositionZText)
    {
        PositionZText->SetText(FText::FromString(FString::Printf(TEXT("%.2f"), Telemetry.Position.Z)));
    }

    // Update velocity
    if (VelocityXText)
    {
        VelocityXText->SetText(FText::FromString(FString::Printf(TEXT("%.2f"), Telemetry.Velocity.X)));
    }
    if (VelocityYText)
    {
        VelocityYText->SetText(FText::FromString(FString::Printf(TEXT("%.2f"), Telemetry.Velocity.Y)));
    }
    if (VelocityZText)
    {
        VelocityZText->SetText(FText::FromString(FString::Printf(TEXT("%.2f"), Telemetry.Velocity.Z)));
    }

    // Update attitude
    if (PitchText)
    {
        PitchText->SetText(FText::FromString(FString::Printf(TEXT("%.1f"), Telemetry.Attitude.Pitch)));
    }
    if (RollText)
    {
        RollText->SetText(FText::FromString(FString::Printf(TEXT("%.1f"), Telemetry.Attitude.Roll)));
    }
    if (YawText)
    {
        YawText->SetText(FText::FromString(FString::Printf(TEXT("%.1f"), Telemetry.Attitude.Yaw)));
    }

    // Update summary values
    if (AltitudeText)
    {
        AltitudeText->SetText(FText::FromString(FString::Printf(TEXT("%.1f m"), Telemetry.AltitudeM)));
    }
    if (SpeedText)
    {
        SpeedText->SetText(FText::FromString(FString::Printf(TEXT("%.1f m/s"), Telemetry.SpeedMs)));
    }

    // Notify Blueprint
    OnTelemetryUpdated(Telemetry);
}
