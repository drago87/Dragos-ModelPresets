/**
 * Dragos Model Presets - Frontend Extension for ComfyUI
 *
 * Provides dynamic UI for the Dragos Model Settings, LoRA Settings (Model+Clip),
 * and LoRA Settings (Model Only) nodes.
 *
 * Features:
 * - Dynamic model name dropdown based on source type
 * - Dynamic saved setting dropdown based on selected model
 * - Save / Save New / Reset / Delete buttons for model presets
 * - Save / Reset buttons for LoRA presets
 * - Multi-select for lora_type
 * - Auto-populate settings from saved presets
 * - Server communication via custom API routes
 */

import { app, api } from "../../scripts/app.js";

// ===========================================================================
// API Helpers
// ===========================================================================

const DRAGOS_API = {
    async getModelsData() {
        const resp = await api.fetchApi("/dragos/data/models");
        return resp.json();
    },
    async getLorasData() {
        const resp = await api.fetchApi("/dragos/data/loras");
        return resp.json();
    },
    async getModelList(sourceType) {
        const resp = await api.fetchApi(`/dragos/model_list?source_type=${encodeURIComponent(sourceType)}`);
        return resp.json();
    },
    async getLoraList() {
        const resp = await api.fetchApi("/dragos/lora_list");
        return resp.json();
    },
    async getModelTypes() {
        const resp = await api.fetchApi("/dragos/model_types");
        return resp.json();
    },
    async getSamplers() {
        const resp = await api.fetchApi("/dragos/samplers");
        return resp.json();
    },
    async getModelSettings(sourceType, modelName) {
        const resp = await api.fetchApi(
            `/dragos/model_settings?source_type=${encodeURIComponent(sourceType)}&model_name=${encodeURIComponent(modelName)}`
        );
        return resp.json();
    },
    async getLoraMetadata(loraName) {
        const resp = await api.fetchApi(`/dragos/lora_metadata?lora_name=${encodeURIComponent(loraName)}`);
        return resp.json();
    },
    async saveModelSetting(data) {
        const resp = await api.fetchApi("/dragos/models/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        return resp.json();
    },
    async saveNewModelSetting(data) {
        const resp = await api.fetchApi("/dragos/models/save_new", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        return resp.json();
    },
    async deleteModelSetting(data) {
        const resp = await api.fetchApi("/dragos/models/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        return resp.json();
    },
    async saveLora(data) {
        const resp = await api.fetchApi("/dragos/loras/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });
        return resp.json();
    },
};

// ===========================================================================
// Widget Helpers
// ===========================================================================

/**
 * Find a widget by name on a node.
 */
function findWidget(node, name) {
    return node.widgets?.find(w => w.name === name);
}

/**
 * Update a combo widget's options and reset value if current value is invalid.
 *
 * Handles ComfyUI's combo widget system properly by:
 * 1. Updating widget.options.values (the data model)
 * 2. Rebuilding the HTML <select> element if present (the UI)
 * 3. Forcing a canvas redraw so the change is visible
 */
function updateComboOptions(widget, newOptions) {
    if (!widget || !newOptions) return;
    const oldValue = widget.value;

    // Update the data model
    widget.options.values = newOptions;

    // Determine the new value
    let newValue;
    if (newOptions.length > 0) {
        newValue = newOptions.includes(oldValue) ? oldValue : newOptions[0];
    } else {
        newValue = "(none)";
    }
    widget.value = newValue;

    // Rebuild the HTML <select> element if the widget has one
    // ComfyUI's combo widgets use an inputEl (a <select>) that must be
    // repopulated when options change, otherwise the UI is stale.
    if (widget.inputEl) {
        const oldSel = widget.inputEl.value;
        widget.inputEl.innerHTML = "";
        for (const opt of newOptions) {
            const o = document.createElement("option");
            o.value = opt;
            o.textContent = opt;
            widget.inputEl.appendChild(o);
        }
        widget.inputEl.value = newValue;
    }

    // Force canvas redraw so the widget visually updates
    try {
        const canvas = app.canvas;
        if (canvas && canvas.setDirty) {
            canvas.setDirty(true);
        }
    } catch (_e) { /* ignore */ }

    // Trigger the callback if the value actually changed
    if (newValue !== oldValue && widget.callback) {
        widget.callback(newValue);
    }
}

/**
 * Set a widget's value and trigger its callback.
 * Also syncs the HTML <select> element for combo widgets.
 */
function setWidgetValue(widget, value, triggerCallback = true) {
    if (!widget) return;
    widget.value = value;
    // Sync the HTML select element if present
    if (widget.inputEl) {
        widget.inputEl.value = value;
    }
    if (triggerCallback && widget.callback) {
        widget.callback(value);
    }
}

// ===========================================================================
// Multi-select Widget for lora_type
// ===========================================================================

/**
 * Create a custom multi-select widget attached to a node.
 * Displays as a text input showing comma-separated values, with a clickable
 * panel that shows checkboxes for each option.
 */
function addMultiSelectWidget(node, name, options, defaultValues = []) {
    // Store selected values as an array in node properties
    if (!node.properties) node.properties = {};
    const propKey = `dragos_${name}`;
    node.properties[propKey] = defaultValues || [];

    // Create a text widget to display the current selection
    const displayWidget = node.addWidget(
        "text",
        name,
        defaultValues.join(", "),
        (v) => {
            // When the text changes, parse it back to array
            if (typeof v === "string") {
                node.properties[propKey] = v.split(",").map(s => s.trim()).filter(s => s.length > 0);
            }
        },
        { multiline: false }
    );

    // Override the draw to make it clear this is a multi-select field
    const originalDraw = displayWidget.draw;
    displayWidget.draw = function(ctx, node, widgetWidth, y, H) {
        if (originalDraw) {
            originalDraw.call(this, ctx, node, widgetWidth, y, H);
        }
    };

    // Add a button to toggle the selection panel
    const btnWidget = node.addWidget("button", `${name} (select)`, null, () => {
        showMultiSelectDialog(node, name, options, propKey, displayWidget);
    });

    // Return the display widget for reference
    return {
        displayWidget,
        btnWidget,
        getValues: () => node.properties[propKey] || [],
        setValues: (vals) => {
            node.properties[propKey] = vals;
            displayWidget.value = vals.join(", ");
        }
    };
}

/**
 * Show a simple multi-select dialog.
 */
function showMultiSelectDialog(node, name, options, propKey, displayWidget) {
    const currentValues = node.properties[propKey] || [];

    // Create dialog
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;";

    const dialog = document.createElement("div");
    dialog.style.cssText = "background:#2a2a2a;border:1px solid #555;border-radius:8px;padding:16px;min-width:300px;max-width:500px;max-height:80vh;overflow-y:auto;color:#ddd;font-family:sans-serif;";

    const title = document.createElement("h3");
    title.textContent = `Select ${name}`;
    title.style.cssText = "margin:0 0 12px 0;color:#fff;";
    dialog.appendChild(title);

    const checkboxContainer = document.createElement("div");
    checkboxContainer.style.cssText = "display:flex;flex-direction:column;gap:4px;";

    const checkboxes = [];
    for (const opt of options) {
        const label = document.createElement("label");
        label.style.cssText = "display:flex;align-items:center;gap:8px;cursor:pointer;padding:4px;";

        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = currentValues.includes(opt);
        cb.value = opt;
        checkboxes.push(cb);

        const span = document.createElement("span");
        span.textContent = opt;
        span.style.cssText = "user-select:none;";

        label.appendChild(cb);
        label.appendChild(span);
        checkboxContainer.appendChild(label);
    }
    dialog.appendChild(checkboxContainer);

    // Buttons
    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;margin-top:16px;justify-content:flex-end;";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = "padding:6px 16px;border:1px solid #555;border-radius:4px;background:#444;color:#ddd;cursor:pointer;";
    cancelBtn.onclick = () => overlay.remove();

    const okBtn = document.createElement("button");
    okBtn.textContent = "OK";
    okBtn.style.cssText = "padding:6px 16px;border:1px solid #555;border-radius:4px;background:#4a9;color:#fff;cursor:pointer;font-weight:bold;";
    okBtn.onclick = () => {
        const selected = checkboxes.filter(cb => cb.checked).map(cb => cb.value);
        node.properties[propKey] = selected;
        displayWidget.value = selected.join(", ");
        overlay.remove();
    };

    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(okBtn);
    dialog.appendChild(btnRow);

    overlay.appendChild(dialog);
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    document.body.appendChild(overlay);
}

// ===========================================================================
// Model Settings Node Extension
// ===========================================================================

function setupModelSettingsNode(nodeType, nodeData) {
    // Cache for server data
    const _cache = {
        modelsData: null,
        modelTypes: null,
    };

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (originalOnNodeCreated) {
            originalOnNodeCreated.call(this);
        }

        const node = this;

        // Store properties for tracking state
        node.properties = node.properties || {};
        node.properties._dragos_last_source = "";
        node.properties._dragos_last_model = "";

        // Add action buttons at the bottom
        const saveBtn = node.addWidget("button", "Save", null, () => {
            handleModelSave(node);
        });
        saveBtn.serialize = false;

        const saveNewBtn = node.addWidget("button", "Save New", null, () => {
            handleModelSaveNew(node);
        });
        saveNewBtn.serialize = false;

        const resetBtn = node.addWidget("button", "Reset", null, () => {
            handleModelReset(node);
        });
        resetBtn.serialize = false;

        const deleteBtn = node.addWidget("button", "Delete", null, () => {
            handleModelDelete(node);
        });
        deleteBtn.serialize = false;

        // Hook into model_source_type change to update model_name dropdown
        const sourceWidget = findWidget(node, "model_source_type");
        const modelWidget = findWidget(node, "model_name");

        if (sourceWidget) {
            const origSourceCallback = sourceWidget.callback;
            sourceWidget.callback = async function (value) {
                if (origSourceCallback) origSourceCallback.call(this, value);
                await updateModelNameDropdown(node, value);
            };
        }

        // Hook into model_name change to update saved_setting dropdown and load metadata
        if (modelWidget) {
            const origModelCallback = modelWidget.callback;
            modelWidget.callback = async function (value) {
                if (origModelCallback) origModelCallback.call(this, value);
                await loadModelMetadata(node);
            };
        }

        // Hook into saved_setting change to populate setting fields
        const settingWidget = findWidget(node, "saved_setting");
        if (settingWidget) {
            const origSettingCallback = settingWidget.callback;
            settingWidget.callback = async function (value) {
                if (origSettingCallback) origSettingCallback.call(this, value);
                await loadSettingValues(node, value);
            };
        }

        // Trigger initial load
        setTimeout(async () => {
            const sourceType = sourceWidget?.value || "checkpoint_models";

            // Fetch the latest model_types list from the server and update the dropdown
            try {
                const mtData = await DRAGOS_API.getModelTypes();
                const mtList = mtData.model_types || [];
                if (mtList.length > 0) {
                    const modelTypeWidget = findWidget(node, "model_type");
                    if (modelTypeWidget) {
                        updateComboOptions(modelTypeWidget, mtList);
                    }
                }
            } catch (e) {
                console.warn("[Dragos] Could not refresh model_type list:", e);
            }

            await updateModelNameDropdown(node, sourceType);
        }, 100);
    };
}

/**
 * Update the model_name dropdown based on the selected source type.
 */
async function updateModelNameDropdown(node, sourceType) {
    const modelWidget = findWidget(node, "model_name");
    if (!modelWidget) return;

    try {
        const data = await DRAGOS_API.getModelList(sourceType);
        const modelList = data.models || [];
        updateComboOptions(modelWidget, modelList);

        // After updating model list, also update the saved_setting dropdown
        await loadModelMetadata(node);
    } catch (e) {
        console.error("[Dragos] Error updating model list:", e);
    }
}

/**
 * Load metadata (model_type, notes, settings list) for the currently selected model.
 */
async function loadModelMetadata(node) {
    const sourceType = findWidget(node, "model_source_type")?.value;
    const modelName = findWidget(node, "model_name")?.value;
    if (!sourceType || !modelName) return;

    const settingWidget = findWidget(node, "saved_setting");
    const modelTypeWidget = findWidget(node, "model_type");
    const notesWidget = findWidget(node, "notes");

    try {
        const data = await DRAGOS_API.getModelSettings(sourceType, modelName);
        const settings = data.settings || {};

        // Update model_type
        if (modelTypeWidget && data.model_type) {
            setWidgetValue(modelTypeWidget, data.model_type, false);
        }

        // Update notes
        if (notesWidget) {
            setWidgetValue(notesWidget, data.notes || "", false);
        }

        // Update saved_setting dropdown
        const settingNames = Object.keys(settings);
        if (settingWidget) {
            if (settingNames.length > 0) {
                updateComboOptions(settingWidget, settingNames);
                // Auto-load the first setting
                await loadSettingValues(node, settingNames[0]);
            } else {
                // No saved settings - show placeholder and load base defaults
                updateComboOptions(settingWidget, ["(none)"]);
                // Populate with base_model_settings defaults
                const modelsData = await DRAGOS_API.getModelsData();
                const base = modelsData.base_model_settings || {};
                const turboWidget = findWidget(node, "turbo_lora");
                const stepsWidget = findWidget(node, "steps");
                const cfgWidget = findWidget(node, "cfg");
                const samplerWidget = findWidget(node, "sampler_name");
                const schedulerWidget = findWidget(node, "scheduler");
                if (turboWidget) setWidgetValue(turboWidget, false, false);
                if (stepsWidget) setWidgetValue(stepsWidget, base.steps ?? 20, false);
                if (cfgWidget) setWidgetValue(cfgWidget, base.cfg ?? 8.0, false);
                if (samplerWidget) setWidgetValue(samplerWidget, base.sampler_name || "euler", false);
                if (schedulerWidget) setWidgetValue(schedulerWidget, base.scheduler || "simple", false);
            }
        }
    } catch (e) {
        console.error("[Dragos] Error loading model metadata:", e);
    }
}

/**
 * Load a specific setting's values into the node's widgets.
 */
async function loadSettingValues(node, settingName) {
    const sourceType = findWidget(node, "model_source_type")?.value;
    const modelName = findWidget(node, "model_name")?.value;
    if (!sourceType || !modelName) return;

    try {
        const data = await DRAGOS_API.getModelSettings(sourceType, modelName);
        const settings = data.settings || {};
        const setting = settings[settingName];

        if (setting) {
            const turboWidget = findWidget(node, "turbo_lora");
            const stepsWidget = findWidget(node, "steps");
            const cfgWidget = findWidget(node, "cfg");
            const samplerWidget = findWidget(node, "sampler_name");
            const schedulerWidget = findWidget(node, "scheduler");

            if (turboWidget) setWidgetValue(turboWidget, setting.turbo_lora || false, false);
            if (stepsWidget) setWidgetValue(stepsWidget, setting.steps ?? 20, false);
            if (cfgWidget) setWidgetValue(cfgWidget, setting.cfg ?? 8.0, false);
            if (samplerWidget) setWidgetValue(samplerWidget, setting.sampler_name || "euler", false);
            if (schedulerWidget) setWidgetValue(schedulerWidget, setting.scheduler || "simple", false);
        }
    } catch (e) {
        console.error("[Dragos] Error loading setting values:", e);
    }
}

/**
 * Handle Save button click - save current values to the selected setting.
 */
async function handleModelSave(node) {
    const sourceType = findWidget(node, "model_source_type")?.value;
    const modelName = findWidget(node, "model_name")?.value;
    const settingName = findWidget(node, "saved_setting")?.value;
    const modelType = findWidget(node, "model_type")?.value;
    const turboLora = findWidget(node, "turbo_lora")?.value;
    const steps = findWidget(node, "steps")?.value;
    const cfg = findWidget(node, "cfg")?.value;
    const samplerName = findWidget(node, "sampler_name")?.value;
    const scheduler = findWidget(node, "scheduler")?.value;
    const notes = findWidget(node, "notes")?.value;

    if (!sourceType || !modelName || !settingName) {
        alert("Please select a model source, model, and setting name before saving.");
        return;
    }

    try {
        const result = await DRAGOS_API.saveModelSetting({
            source_type: sourceType,
            model_name: modelName,
            setting_name: settingName,
            model_type: modelType,
            turbo_lora: turboLora,
            steps: steps,
            cfg: cfg,
            sampler_name: samplerName,
            scheduler: scheduler,
            notes: notes,
        });

        if (result.error) {
            alert(`Error saving: ${result.error}`);
        } else {
            // Refresh the settings dropdown
            await loadModelMetadata(node);
        }
    } catch (e) {
        console.error("[Dragos] Error saving model setting:", e);
        alert("Error saving model setting. Check console for details.");
    }
}

/**
 * Handle Save New button click - create a new named setting.
 */
async function handleModelSaveNew(node) {
    const sourceType = findWidget(node, "model_source_type")?.value;
    const modelName = findWidget(node, "model_name")?.value;
    const modelType = findWidget(node, "model_type")?.value;
    const turboLora = findWidget(node, "turbo_lora")?.value;
    const steps = findWidget(node, "steps")?.value;
    const cfg = findWidget(node, "cfg")?.value;
    const samplerName = findWidget(node, "sampler_name")?.value;
    const scheduler = findWidget(node, "scheduler")?.value;
    const notes = findWidget(node, "notes")?.value;

    if (!sourceType || !modelName) {
        alert("Please select a model source and model before saving a new setting.");
        return;
    }

    const newSettingName = prompt("Enter a name for the new setting:");
    if (!newSettingName || !newSettingName.trim()) return;

    try {
        const result = await DRAGOS_API.saveNewModelSetting({
            source_type: sourceType,
            model_name: modelName,
            new_setting_name: newSettingName.trim(),
            model_type: modelType,
            turbo_lora: turboLora,
            steps: steps,
            cfg: cfg,
            sampler_name: samplerName,
            scheduler: scheduler,
            notes: notes,
        });

        if (result.error) {
            alert(`Error: ${result.error}`);
        } else {
            // Refresh and select the new setting
            await loadModelMetadata(node);
            const settingWidget = findWidget(node, "saved_setting");
            if (settingWidget) {
                setWidgetValue(settingWidget, newSettingName.trim());
            }
        }
    } catch (e) {
        console.error("[Dragos] Error saving new model setting:", e);
        alert("Error saving new model setting. Check console for details.");
    }
}

/**
 * Handle Reset button click - reload current setting from disk.
 */
async function handleModelReset(node) {
    const settingName = findWidget(node, "saved_setting")?.value;
    if (!settingName || settingName === "(none)") {
        // Reset to base settings
        try {
            const data = await DRAGOS_API.getModelsData();
            const base = data.base_model_settings || {};
            const turboWidget = findWidget(node, "turbo_lora");
            const stepsWidget = findWidget(node, "steps");
            const cfgWidget = findWidget(node, "cfg");
            const samplerWidget = findWidget(node, "sampler_name");
            const schedulerWidget = findWidget(node, "scheduler");

            if (turboWidget) setWidgetValue(turboWidget, false, false);
            if (stepsWidget) setWidgetValue(stepsWidget, base.steps ?? 20, false);
            if (cfgWidget) setWidgetValue(cfgWidget, base.cfg ?? 8.0, false);
            if (samplerWidget) setWidgetValue(samplerWidget, base.sampler_name || "euler", false);
            if (schedulerWidget) setWidgetValue(schedulerWidget, base.scheduler || "simple", false);
        } catch (e) {
            console.error("[Dragos] Error resetting to base settings:", e);
        }
        return;
    }

    await loadSettingValues(node, settingName);
}

/**
 * Handle Delete button click - delete the currently selected setting.
 */
async function handleModelDelete(node) {
    const sourceType = findWidget(node, "model_source_type")?.value;
    const modelName = findWidget(node, "model_name")?.value;
    const settingName = findWidget(node, "saved_setting")?.value;

    if (!sourceType || !modelName || !settingName) {
        alert("No setting selected to delete.");
        return;
    }

    if (!confirm(`Are you sure you want to delete the setting "${settingName}" for ${modelName}?`)) {
        return;
    }

    try {
        const result = await DRAGOS_API.deleteModelSetting({
            source_type: sourceType,
            model_name: modelName,
            setting_name: settingName,
        });

        if (result.error) {
            alert(`Error: ${result.error}`);
        } else {
            // Refresh the settings
            await loadModelMetadata(node);
        }
    } catch (e) {
        console.error("[Dragos] Error deleting model setting:", e);
        alert("Error deleting model setting. Check console for details.");
    }
}


// ===========================================================================
// LoRA Settings Node Extension (shared between Model+Clip and Model Only)
// ===========================================================================

function setupLoraSettingsNode(nodeType, nodeData, isModelOnly) {
    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (originalOnNodeCreated) {
            originalOnNodeCreated.call(this);
        }

        const node = this;
        node.properties = node.properties || {};

        // Add multi-select for lora_type
        const modelTypesCache = { types: [] };

        // Fetch model types list asynchronously and add the multi-select widget
        DRAGOS_API.getModelTypes().then(data => {
            modelTypesCache.types = data.model_types || [];
            addMultiSelectWidget(node, "lora_type", modelTypesCache.types, []);
        }).catch(e => {
            console.error("[Dragos] Error fetching model types for lora_type:", e);
        });

        // Add action buttons
        const saveBtn = node.addWidget("button", "Save", null, () => {
            handleLoraSave(node, isModelOnly);
        });
        saveBtn.serialize = false;

        const resetBtn = node.addWidget("button", "Reset", null, () => {
            handleLoraReset(node, isModelOnly);
        });
        resetBtn.serialize = false;

        // Hook into lora_name change to load metadata
        const loraWidget = findWidget(node, "lora_name");
        if (loraWidget) {
            const origLoraCallback = loraWidget.callback;
            loraWidget.callback = async function (value) {
                if (origLoraCallback) origLoraCallback.call(this, value);
                await loadLoraMetadata(node, isModelOnly);
            };
        }
    };
}

/**
 * Load metadata for the selected LoRA.
 */
async function loadLoraMetadata(node, isModelOnly) {
    const loraName = findWidget(node, "lora_name")?.value;
    if (!loraName) return;

    try {
        const metadata = await DRAGOS_API.getLoraMetadata(loraName);

        // Update widgets
        const turboWidget = findWidget(node, "turbo_lora");
        const strengthModelWidget = findWidget(node, "strength_model");
        const strengthClipWidget = findWidget(node, "strength_clip");
        const notesWidget = findWidget(node, "notes");

        if (turboWidget) setWidgetValue(turboWidget, metadata.turbo_lora || false, false);
        if (strengthModelWidget) setWidgetValue(strengthModelWidget, metadata.strength_model ?? 1.00, false);
        if (!isModelOnly && strengthClipWidget) {
            setWidgetValue(strengthClipWidget, metadata.strength_clip ?? 1.00, false);
        }
        if (notesWidget) setWidgetValue(notesWidget, metadata.notes || "", false);

        // Update lora_type multi-select
        const propKey = "dragos_lora_type";
        if (node.properties) {
            node.properties[propKey] = metadata.lora_type || [];
            // Update the display widget if it exists
            const loraTypeDisplay = node.widgets?.find(w => w.name === "lora_type");
            if (loraTypeDisplay) {
                loraTypeDisplay.value = (metadata.lora_type || []).join(", ");
            }
        }
    } catch (e) {
        console.error("[Dragos] Error loading LoRA metadata:", e);
    }
}

/**
 * Handle Save button click for LoRA nodes.
 */
async function handleLoraSave(node, isModelOnly) {
    const loraName = findWidget(node, "lora_name")?.value;
    if (!loraName) {
        alert("Please select a LoRA before saving.");
        return;
    }

    const turboLora = findWidget(node, "turbo_lora")?.value;
    const strengthModel = findWidget(node, "strength_model")?.value;
    const strengthClip = isModelOnly ? undefined : findWidget(node, "strength_clip")?.value;
    const notes = findWidget(node, "notes")?.value;
    const loraType = node.properties?.dragos_lora_type || [];

    try {
        const payload = {
            lora_name: loraName,
            lora_type: loraType,
            turbo_lora: turboLora,
            strength_model: strengthModel,
            notes: notes,
        };

        // Only include strength_clip for Model+Clip node
        if (!isModelOnly && strengthClip !== undefined) {
            payload.strength_clip = strengthClip;
        }

        const result = await DRAGOS_API.saveLora(payload);

        if (result.error) {
            alert(`Error saving: ${result.error}`);
        }
    } catch (e) {
        console.error("[Dragos] Error saving LoRA:", e);
        alert("Error saving LoRA. Check console for details.");
    }
}

/**
 * Handle Reset button click for LoRA nodes.
 */
async function handleLoraReset(node, isModelOnly) {
    await loadLoraMetadata(node, isModelOnly);
}


// ===========================================================================
// Extension Registration
// ===========================================================================

app.registerExtension({
    name: "dragos.modelpresets",

    async setup() {
        // Pre-fetch data that all nodes might need
        try {
            await DRAGOS_API.getModelTypes();
        } catch (e) {
            console.warn("[Dragos] Could not pre-fetch model types:", e);
        }
    },

    beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        const name = nodeData.name;

        // Use internal node names from NODE_CLASS_MAPPINGS
        if (name === "DragosModelSettings") {
            setupModelSettingsNode(nodeType, nodeData);
        }
        else if (name === "DragosLoraSettingsModelClip") {
            setupLoraSettingsNode(nodeType, nodeData, false);
        }
        else if (name === "DragosLoraSettingsModelOnly") {
            setupLoraSettingsNode(nodeType, nodeData, true);
        }
    },

    // Handle node serialization - preserve custom properties
    onNodeConfigure(node) {
        // Ensure lora_type property is preserved during graph save/load
        if (node.properties && node.properties.dragos_lora_type) {
            const loraTypeDisplay = node.widgets?.find(w => w.name === "lora_type");
            if (loraTypeDisplay) {
                loraTypeDisplay.value = node.properties.dragos_lora_type.join(", ");
            }
        }
    },
});
