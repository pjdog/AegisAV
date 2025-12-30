// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVCameraPanel.h"
#include "Components/Image.h"
#include "Components/TextBlock.h"
#include "Materials/MaterialInstanceDynamic.h"
#include "Engine/Texture2D.h"

UAegisAVCameraPanel::UAegisAVCameraPanel(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
    , CurrentFrame(nullptr)
    , DynamicMaterial(nullptr)
    , FrameCount(0)
    , LastFrameTime(0.0f)
{
    PanelTitle = FText::FromString(TEXT("Camera Feed"));
    ExpandedSize = FVector2D(400.0f, 280.0f);
}

void UAegisAVCameraPanel::NativeConstruct()
{
    Super::NativeConstruct();

    // Create dynamic material if base material is set
    CreateDynamicMaterial();

    // Set default label
    if (CameraLabel)
    {
        CameraLabel->SetText(FText::FromString(TEXT("No Stream")));
    }

    if (FrameInfoText)
    {
        FrameInfoText->SetText(FText::FromString(TEXT("0 fps")));
    }
}

void UAegisAVCameraPanel::CreateDynamicMaterial()
{
    if (CameraMaterial && CameraImage)
    {
        DynamicMaterial = UMaterialInstanceDynamic::Create(CameraMaterial, this);
        if (DynamicMaterial)
        {
            CameraImage->SetBrushFromMaterial(DynamicMaterial);
        }
    }
}

void UAegisAVCameraPanel::UpdateCameraFrame(UTexture2D* NewFrame)
{
    if (!NewFrame)
    {
        return;
    }

    CurrentFrame = NewFrame;
    FrameCount++;

    // Update the image display
    if (CameraImage)
    {
        if (DynamicMaterial)
        {
            // Update texture parameter in dynamic material
            DynamicMaterial->SetTextureParameterValue(FName("CameraTexture"), NewFrame);
        }
        else
        {
            // Direct texture brush (simpler approach)
            CameraImage->SetBrushFromTexture(NewFrame, true);
        }
    }

    // Calculate and display FPS
    float CurrentTime = FPlatformTime::Seconds();
    if (LastFrameTime > 0.0f && FrameInfoText)
    {
        float DeltaTime = CurrentTime - LastFrameTime;
        if (DeltaTime > 0.0f)
        {
            float FPS = 1.0f / DeltaTime;
            FrameInfoText->SetText(FText::FromString(FString::Printf(TEXT("%.1f fps | %dx%d"),
                FPS, NewFrame->GetSizeX(), NewFrame->GetSizeY())));
        }
    }
    LastFrameTime = CurrentTime;

    // Notify Blueprint
    OnCameraFrameUpdated(NewFrame);
}

void UAegisAVCameraPanel::SetCameraLabel(const FString& Label)
{
    if (CameraLabel)
    {
        CameraLabel->SetText(FText::FromString(Label));
    }
}
