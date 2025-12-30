// Copyright AegisAV Team. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Blueprint/UserWidget.h"
#include "AegisAVBasePanel.generated.h"

class UBorder;
class UTextBlock;
class UButton;
class UNamedSlot;
class UCanvasPanel;
class UCanvasPanelSlot;

/**
 * Base class for draggable/collapsible AegisAV panels.
 *
 * Features:
 * - Drag to move by clicking and dragging the title bar
 * - Collapse/expand by clicking the collapse button
 * - Bring to front when clicked
 * - Clamp to viewport bounds
 *
 * Blueprint Usage:
 * - Create a Widget Blueprint that inherits from this class
 * - Add widgets with specific names that will be auto-bound:
 *   - TitleBar (UBorder): The draggable header area
 *   - TitleText (UTextBlock): Displays the panel title
 *   - CollapseButton (UButton): Toggles collapsed state
 *   - ContentSlot (UNamedSlot): Container for panel content
 *
 * Example Widget Hierarchy:
 * [Canvas Panel]
 *   +- [Border] (Name: TitleBar)
 *   |    +- [Horizontal Box]
 *   |         +- [Text Block] (Name: TitleText)
 *   |         +- [Spacer]
 *   |         +- [Button] (Name: CollapseButton)
 *   +- [Named Slot] (Name: ContentSlot)
 */
UCLASS(Abstract, Blueprintable)
class AEGISAVOVERLAY_API UAegisAVBasePanel : public UUserWidget
{
    GENERATED_BODY()

public:
    UAegisAVBasePanel(const FObjectInitializer& ObjectInitializer);

    // ========================================================================
    // Panel Configuration
    // ========================================================================

    /** Title displayed in the panel header */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panel")
    FText PanelTitle;

    /** Whether this panel can be collapsed */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panel")
    bool bIsCollapsible = true;

    /** Whether this panel can be dragged */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panel")
    bool bIsDraggable = true;

    /** Size when minimized (collapsed) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panel")
    FVector2D MinimizedSize = FVector2D(250.0f, 40.0f);

    /** Size when expanded */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Panel")
    FVector2D ExpandedSize = FVector2D(350.0f, 250.0f);

    // ========================================================================
    // Panel State
    // ========================================================================

    /** Current collapsed state */
    UPROPERTY(BlueprintReadOnly, Category = "Panel")
    bool bIsCollapsed = false;

    /** Currently being dragged */
    UPROPERTY(BlueprintReadOnly, Category = "Panel")
    bool bIsDragging = false;

    // ========================================================================
    // Actions
    // ========================================================================

    /** Toggle between collapsed and expanded */
    UFUNCTION(BlueprintCallable, Category = "Panel")
    void ToggleCollapse();

    /** Set collapsed state */
    UFUNCTION(BlueprintCallable, Category = "Panel")
    void SetCollapsed(bool bCollapse);

    /** Bring this panel to the front (highest Z-order) */
    UFUNCTION(BlueprintCallable, Category = "Panel")
    void BringToFront();

    /** Set the panel position */
    UFUNCTION(BlueprintCallable, Category = "Panel")
    void SetPanelPosition(FVector2D NewPosition);

    /** Get the current panel position */
    UFUNCTION(BlueprintPure, Category = "Panel")
    FVector2D GetPanelPosition() const { return CurrentPosition; }

    // ========================================================================
    // Events
    // ========================================================================

    /** Called when collapsed state changes */
    UFUNCTION(BlueprintImplementableEvent, Category = "Panel|Events")
    void OnCollapsedStateChanged(bool bNewCollapsed);

    /** Called when drag starts */
    UFUNCTION(BlueprintImplementableEvent, Category = "Panel|Events")
    void OnDragStarted();

    /** Called when drag ends */
    UFUNCTION(BlueprintImplementableEvent, Category = "Panel|Events")
    void OnDragEnded();

protected:
    // ========================================================================
    // Widget Bindings (set in Blueprint via BindWidget)
    // ========================================================================

    /** The draggable title bar border */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UBorder* TitleBar;

    /** The title text display */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UTextBlock* TitleText;

    /** Button to toggle collapse */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UButton* CollapseButton;

    /** Named slot for panel content */
    UPROPERTY(BlueprintReadOnly, meta = (BindWidgetOptional))
    UNamedSlot* ContentSlot;

    // ========================================================================
    // Overrides
    // ========================================================================

    virtual void NativeConstruct() override;
    virtual void NativeDestruct() override;

    virtual FReply NativeOnMouseButtonDown(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) override;
    virtual FReply NativeOnMouseButtonUp(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) override;
    virtual FReply NativeOnMouseMove(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) override;
    virtual void NativeOnMouseLeave(const FPointerEvent& InMouseEvent) override;

private:
    // ========================================================================
    // Internal State
    // ========================================================================

    /** Offset from mouse to widget origin when dragging */
    FVector2D DragOffset;

    /** Current position of the panel */
    FVector2D CurrentPosition;

    /** Static Z-order counter for bring-to-front */
    static int32 GlobalZOrderCounter;

    // ========================================================================
    // Internal Methods
    // ========================================================================

    /** Check if mouse is over the title bar area */
    bool IsOverTitleBar(const FGeometry& InGeometry, const FPointerEvent& InMouseEvent) const;

    /** Update the panel's position on screen */
    void UpdatePanelPosition(const FVector2D& NewPosition);

    /** Clamp position to keep panel visible on screen */
    FVector2D ClampToViewport(const FVector2D& Position) const;

    /** Handle collapse button click */
    UFUNCTION()
    void OnCollapseButtonClicked();
};
