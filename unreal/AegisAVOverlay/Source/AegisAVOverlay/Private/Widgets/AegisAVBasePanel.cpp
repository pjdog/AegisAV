// Copyright AegisAV Team. All Rights Reserved.

#include "Widgets/AegisAVBasePanel.h"
#include "AegisAVOverlay.h"
#include "Components/Border.h"
#include "Components/TextBlock.h"
#include "Components/Button.h"
#include "Components/NamedSlot.h"
#include "Components/CanvasPanel.h"
#include "Components/CanvasPanelSlot.h"
#include "Blueprint/WidgetLayoutLibrary.h"
#include "Engine/Engine.h"
#include "Engine/GameViewportClient.h"

// Static member initialization
int32 UAegisAVBasePanel::GlobalZOrderCounter = 100;

UAegisAVBasePanel::UAegisAVBasePanel(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
    , PanelTitle(FText::FromString(TEXT("Panel")))
    , bIsCollapsible(true)
    , bIsDraggable(true)
    , MinimizedSize(250.0f, 40.0f)
    , ExpandedSize(350.0f, 250.0f)
    , bIsCollapsed(false)
    , bIsDragging(false)
    , DragOffset(FVector2D::ZeroVector)
    , CurrentPosition(FVector2D::ZeroVector)
{
}

void UAegisAVBasePanel::NativeConstruct()
{
    Super::NativeConstruct();

    // Set up title text
    if (TitleText)
    {
        TitleText->SetText(PanelTitle);
    }

    // Bind collapse button
    if (CollapseButton && bIsCollapsible)
    {
        CollapseButton->OnClicked.AddDynamic(this, &UAegisAVBasePanel::OnCollapseButtonClicked);
    }
    else if (CollapseButton && !bIsCollapsible)
    {
        CollapseButton->SetVisibility(ESlateVisibility::Collapsed);
    }
}

void UAegisAVBasePanel::NativeDestruct()
{
    if (CollapseButton)
    {
        CollapseButton->OnClicked.RemoveDynamic(this, &UAegisAVBasePanel::OnCollapseButtonClicked);
    }

    Super::NativeDestruct();
}

FReply UAegisAVBasePanel::NativeOnMouseButtonDown(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent)
{
    // Bring to front on any click
    BringToFront();

    // Check for left mouse button
    if (InMouseEvent.GetEffectingButton() == EKeys::LeftMouseButton)
    {
        // Check if we're over the title bar for dragging
        if (bIsDraggable && IsOverTitleBar(InGeometry, InMouseEvent))
        {
            bIsDragging = true;

            // Calculate offset from widget origin to mouse position
            FVector2D LocalPos = InGeometry.AbsoluteToLocal(InMouseEvent.GetScreenSpacePosition());
            DragOffset = LocalPos;

            OnDragStarted();

            return FReply::Handled().CaptureMouse(TakeWidget());
        }
    }

    return FReply::Unhandled();
}

FReply UAegisAVBasePanel::NativeOnMouseButtonUp(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent)
{
    if (bIsDragging && InMouseEvent.GetEffectingButton() == EKeys::LeftMouseButton)
    {
        bIsDragging = false;
        OnDragEnded();
        return FReply::Handled().ReleaseMouseCapture();
    }

    return FReply::Unhandled();
}

FReply UAegisAVBasePanel::NativeOnMouseMove(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent)
{
    if (bIsDragging)
    {
        // Get screen position of mouse
        FVector2D MouseScreenPos = InMouseEvent.GetScreenSpacePosition();

        // Calculate new widget position
        FVector2D NewPosition = MouseScreenPos - DragOffset;

        // Clamp to viewport
        NewPosition = ClampToViewport(NewPosition);

        // Update position
        UpdatePanelPosition(NewPosition);

        return FReply::Handled();
    }

    return FReply::Unhandled();
}

void UAegisAVBasePanel::NativeOnMouseLeave(const FPointerEvent& InMouseEvent)
{
    // Don't cancel drag when mouse leaves - we have mouse capture
    Super::NativeOnMouseLeave(InMouseEvent);
}

bool UAegisAVBasePanel::IsOverTitleBar(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) const
{
    if (TitleBar)
    {
        // Get the title bar's geometry
        const FGeometry& TitleBarGeometry = TitleBar->GetCachedGeometry();

        // Check if mouse is within title bar bounds
        FVector2D LocalPos = TitleBarGeometry.AbsoluteToLocal(InMouseEvent.GetScreenSpacePosition());
        FVector2D Size = TitleBarGeometry.GetLocalSize();

        return LocalPos.X >= 0 && LocalPos.X <= Size.X &&
               LocalPos.Y >= 0 && LocalPos.Y <= Size.Y;
    }

    // If no title bar specified, use the top portion of the widget
    FVector2D LocalPos = InGeometry.AbsoluteToLocal(InMouseEvent.GetScreenSpacePosition());
    return LocalPos.Y <= MinimizedSize.Y; // Title bar height approximation
}

void UAegisAVBasePanel::UpdatePanelPosition(const FVector2D& NewPosition)
{
    CurrentPosition = NewPosition;

    // Try to update via canvas slot first (preferred method)
    if (UCanvasPanelSlot* CanvasSlot = Cast<UCanvasPanelSlot>(Slot))
    {
        CanvasSlot->SetPosition(NewPosition);
    }
    else
    {
        // Fallback: use render transform
        SetRenderTranslation(NewPosition);
    }
}

FVector2D UAegisAVBasePanel::ClampToViewport(const FVector2D& Position) const
{
    FVector2D ViewportSize = FVector2D::ZeroVector;

    if (GEngine && GEngine->GameViewport)
    {
        GEngine->GameViewport->GetViewportSize(ViewportSize);
    }

    if (ViewportSize.IsNearlyZero())
    {
        // Fallback viewport size
        ViewportSize = FVector2D(1920.0f, 1080.0f);
    }

    // Get current widget size
    FVector2D WidgetSize = bIsCollapsed ? MinimizedSize : ExpandedSize;

    // Clamp position so widget stays within viewport
    // Allow partial off-screen (keep at least title bar visible)
    float MinVisible = 50.0f; // Minimum visible width/height

    FVector2D ClampedPosition;
    ClampedPosition.X = FMath::Clamp(Position.X, -WidgetSize.X + MinVisible, ViewportSize.X - MinVisible);
    ClampedPosition.Y = FMath::Clamp(Position.Y, 0.0f, ViewportSize.Y - MinimizedSize.Y); // Keep title bar on screen

    return ClampedPosition;
}

void UAegisAVBasePanel::ToggleCollapse()
{
    SetCollapsed(!bIsCollapsed);
}

void UAegisAVBasePanel::SetCollapsed(bool bCollapse)
{
    if (!bIsCollapsible || bIsCollapsed == bCollapse)
    {
        return;
    }

    bIsCollapsed = bCollapse;

    // Hide/show content
    if (ContentSlot)
    {
        ContentSlot->SetVisibility(bCollapse ? ESlateVisibility::Collapsed : ESlateVisibility::Visible);
    }

    // Update widget size (if using explicit sizing)
    // Note: In Blueprint, you may want to animate this transition

    // Notify Blueprint
    OnCollapsedStateChanged(bCollapse);
}

void UAegisAVBasePanel::BringToFront()
{
    // Increment global Z-order counter
    GlobalZOrderCounter++;

    // Set this widget's Z-order
    if (UCanvasPanelSlot* CanvasSlot = Cast<UCanvasPanelSlot>(Slot))
    {
        CanvasSlot->SetZOrder(GlobalZOrderCounter);
    }
}

void UAegisAVBasePanel::SetPanelPosition(FVector2D NewPosition)
{
    NewPosition = ClampToViewport(NewPosition);
    UpdatePanelPosition(NewPosition);
}

void UAegisAVBasePanel::OnCollapseButtonClicked()
{
    ToggleCollapse();
}
