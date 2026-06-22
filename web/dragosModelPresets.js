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

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

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
    async rebuildModelIndex() {
        const resp = await api.fetchApi("/dragos/rebuild_model_index", {
            method: "POST",
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
 * Update a combo widget's options and value in-place.
 *
 * This preserves all widget metadata (serialization flags, sizing,
 * custom ComfyUI properties) that would be lost if the widget were
 * replaced via splice + addWidget.
 *
 * @param {object} widget - The combo widget to update
 * @param {string[]} values - The new list of options
 * @param {string} selected - The value to select
 */
function updateComboWidget(widget, values, selected) {
    if (!widget) return;
    widget.options.values = values;
    widget.value = selected;
    app.graph.setDirtyCanvas(true, true);
}

/**
 * Set a widget's value and optionally trigger its callback.
 */
function setWidgetValue(widget, value, triggerCallback = true) {
    if (!widget) return;
    widget.value = value;
    if (triggerCallback && widget.callback) {
        widget.callback(value);
    }
}

/**
 * Show a custom dialog to prompt for a setting name.
 * Replaces browser prompt() which can be blocked in ComfyUI's context.
 * Returns a Promise that resolves to the entered name or null if cancelled.
 */
function promptForSettingName() {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.5);z-index:9999;display:flex;align-items:center;justify-content:center;";

        const dialog = document.createElement("div");
        dialog.style.cssText = "background:#2a2a2a;border:1px solid #555;border-radius:8px;padding:16px;min-width:300px;color:#ddd;font-family:sans-serif;";

        const title = document.createElement("h3");
        title.textContent = "Enter Setting Name";
        title.style.cssText = "margin:0 0 12px 0;color:#fff;";
        dialog.appendChild(title);

        const input = document.createElement("input");
        input.type = "text";
        input.style.cssText = "width:100%;padding:8px;border:1px solid #555;border-radius:4px;background:#333;color:#ddd;font-size:14px;box-sizing:border-box;";
        input.placeholder = "Setting name...";
        dialog.appendChild(input);

        const btnRow = document.createElement("div");
        btnRow.style.cssText = "display:flex;gap:8px;margin-top:12px;justify-content:flex-end;";

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = "padding:6px 16px;border:1px solid #555;border-radius:4px;background:#444;color:#ddd;cursor:pointer;";
        cancelBtn.onclick = () => { overlay.remove(); resolve(null); };

        const okBtn = document.createElement("button");
        okBtn.textContent = "Save";
        okBtn.style.cssText = "padding:6px 16px;border:1px solid #555;border-radius:4px;background:#4a9;color:#fff;cursor:pointer;font-weight:bold;";
        okBtn.onclick = () => {
            const val = input.value.trim();
            if (val) {
                overlay.remove();
                resolve(val);
            }
        };

        btnRow.appendChild(cancelBtn);
        btnRow.appendChild(okBtn);
        dialog.appendChild(btnRow);

        overlay.appendChild(dialog);
        overlay.onclick = (e) => { if (e.target === overlay) { overlay.remove(); resolve(null); } };
        document.body.appendChild(overlay);

        // Focus the input after the dialog is rendered
        setTimeout(() => input.focus(), 50);

        // Handle Enter and Escape keys
        input.onkeydown = (e) => {
            if (e.key === "Enter") {
                const val = input.value.trim();
                if (val) { overlay.remove(); resolve(val); }
            } else if (e.key === "Escape") {
                overlay.remove();
                resolve(null);
            }
        };
    });
}

function addModelSelectorWidget(node) {
    if (node._dragosModelSelectorAdded) {
        return;
    }

    node._dragosModelSelectorAdded = true;

    const modelWidget = findWidget(node, "model_name");

    if (!modelWidget) {
        return;
    }

    // Hide original STRING widget
    modelWidget.hidden = true;

    const BOX_HEIGHT = 30;
	
	const displayWidget = {
		type: "customtext",
		name: "selected_model",
		value: modelWidget.value || "",
	
		draw(ctx, node, width, y, height) {
	
			ctx.save();
	
			ctx.fillStyle = "#222";
			ctx.fillRect(
				10,
				y,
				width - 20,
				BOX_HEIGHT - 4
			);
	
			ctx.strokeStyle = "#555";
			ctx.strokeRect(
				10,
				y,
				width - 20,
				BOX_HEIGHT - 4
			);
	
			ctx.fillStyle = "#DDD";
			ctx.textAlign = "center";
			ctx.textBaseline = "middle";
	
			ctx.fillText(
				this.value || "(no model selected)",
				width / 2,
				y + (BOX_HEIGHT / 2)
			);
	
			ctx.restore();
		},
	
		computeSize(width) {
			return [width, BOX_HEIGHT];
		}
	};

    const displayIndex = node.widgets.findIndex(
        w => w.name === "model_type"
    );

    node.widgets.splice(displayIndex, 0, displayWidget);

    const buttonWidget = {
        type: "button",
        name: "Select Model",
        value: null,
        callback: () => {
            showModelSelectionDialog(
                node,
                modelWidget,
                displayWidget
            );
        },
        computeSize(width) {
            return [width, 32];
        }
    };

    node.widgets.splice(displayIndex + 1, 0, buttonWidget);

    node._dragosModelDisplayWidget = displayWidget;
}

function showModelSelectionDialog(
    node,
    modelWidget,
    displayWidget
) {
    const options = node._dragosModelList || [];

    const overlay = document.createElement("div");

    overlay.style.cssText = `
        position:fixed;
        top:0;
        left:0;
        width:100%;
        height:100%;
        background:rgba(0,0,0,.5);
        z-index:9999;
        display:flex;
        align-items:center;
        justify-content:center;
    `;

    const dialog = document.createElement("div");

    dialog.style.cssText = `
        background:#2a2a2a;
        border:1px solid #555;
        border-radius:8px;
        width:700px;
        max-width:90vw;
        max-height:80vh;
        overflow:auto;
        padding:16px;
        color:#ddd;
    `;

    const select = document.createElement("select");

    select.size = 20;
    select.style.width = "100%";

    for (const option of options) {
        const el = document.createElement("option");

        el.value = option;
        el.textContent = option;

        if (option === modelWidget.value) {
            el.selected = true;
        }

        select.appendChild(el);
    }

    dialog.appendChild(select);

    const buttons = document.createElement("div");

    buttons.style.cssText =
        "display:flex;justify-content:flex-end;margin-top:12px;";

    const ok = document.createElement("button");

    ok.textContent = "OK";

    ok.onclick = async () => {
		modelWidget.value = select.value;
	
		displayWidget.value = select.value;
	
		node.setDirtyCanvas(true, true);
		app.graph.setDirtyCanvas(true, true);
	
		overlay.remove();
	
		await loadModelMetadata(node);
	};

    buttons.appendChild(ok);

    dialog.appendChild(buttons);

    overlay.appendChild(dialog);

    document.body.appendChild(overlay);
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
    if (!node.properties) node.properties = {};

    const propKey = `dragos_${name}`;
    node.properties[propKey] = defaultValues || [];

    const BOX_HEIGHT = 30;

    // Read-only display widget
    const displayWidget = {
        type: "customtext",
        name,
        value: defaultValues.join(", "),

        draw(ctx, node, width, y, height) {
            const text = this.value || "";

            ctx.save();

            ctx.fillStyle = "#222";
            ctx.fillRect(
                10,
                y,
                width - 20,
                BOX_HEIGHT - 4
            );

            ctx.strokeStyle = "#555";
            ctx.strokeRect(
                10,
                y,
                width - 20,
                BOX_HEIGHT - 4
            );

            ctx.fillStyle = "#DDD";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            ctx.fillText(
                text,
                width / 2,
                y + (BOX_HEIGHT / 2)
            );

            ctx.restore();
        },

        computeSize(width) {
            return [width, BOX_HEIGHT];
        }
    };

    node.addCustomWidget(displayWidget);

    const btnWidget = node.addWidget(
        "button",
        "Select Compatible Models",
        null,
        () => {
            showMultiSelectDialog(
                node,
                name,
                options,
                propKey,
                displayWidget
            );
        }
    );

    // Move both widgets above turbo_lora
    const turboIndex = node.widgets.findIndex(
        w => w.name === "turbo_lora"
    );

    if (turboIndex !== -1) {
        const btnIndex = node.widgets.indexOf(btnWidget);
        if (btnIndex !== -1) {
            node.widgets.splice(btnIndex, 1);
        }

        const displayIndex = node.widgets.indexOf(displayWidget);
        if (displayIndex !== -1) {
            node.widgets.splice(displayIndex, 1);
        }

        node.widgets.splice(turboIndex, 0, displayWidget);
        node.widgets.splice(turboIndex + 1, 0, btnWidget);
    }

    return {
        displayWidget,
        btnWidget,

        getValues: () => node.properties[propKey] || [],

        setValues: (vals) => {
            vals = vals || [];

            node.properties[propKey] = vals;
            displayWidget.value = vals.join(", ");

            node.setDirtyCanvas(true, true);
            app.graph.setDirtyCanvas(true, true);
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
	dialog.style.cssText = `
		background:#2a2a2a;
		border:1px solid #555;
		border-radius:8px;
		padding:16px;
		min-width:300px;
		max-width:500px;
		max-height:80vh;
		display:flex;
		flex-direction:column;
		color:#ddd;
		font-family:sans-serif;
	`;

    const title = document.createElement("h3");
    title.textContent = `Select ${name}`;
    title.style.cssText = "margin:0 0 12px 0;color:#fff;";
    dialog.appendChild(title);

    const checkboxContainer = document.createElement("div");
	checkboxContainer.style.cssText = `
		display:flex;
		flex-direction:column;
		gap:4px;
		overflow-y:auto;
		max-height:60vh;
	`;

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
	btnRow.style.cssText = `
		display:flex;
		gap:8px;
		justify-content:flex-end;
		position:sticky;
		bottom:0;
		margin-top:16px;
		padding-top:12px;
		background:#2a2a2a;
		border-top:1px solid #555;
	`;

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
	
		if (displayWidget) {
			displayWidget.value = selected.join(", ");
		}
	
		node.setDirtyCanvas(true, true);
		app.graph.setDirtyCanvas(true, true);
	
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

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (originalOnNodeCreated) {
            originalOnNodeCreated.call(this);
        }

        const node = this;

        // Store properties for tracking state
		node.properties = node.properties || {};
		
		// DEBUG
		console.log(
			"[Dragos] model_name widget:",
			findWidget(node, "model_name")
		);
		
		// Create custom model selector
		addModelSelectorWidget(node);
		
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

        if (sourceWidget) {
            const origSourceCallback = sourceWidget.callback;
            sourceWidget.callback = async function (value) {
                if (origSourceCallback) origSourceCallback.call(this, value);
                await updateModelNameDropdown(node, value);
            };
        }

        // Hook into model_name change to update saved_setting dropdown and load metadata
        const modelWidget = findWidget(node, "model_name");
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
                        const currentModelType = modelTypeWidget.value || "Other";
                        const defaultType = mtList.includes(currentModelType) ? currentModelType : "Other";
                        updateComboWidget(modelTypeWidget, mtList, defaultType);
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
 * Updates the widget in-place to preserve all metadata.
 */
async function updateModelNameDropdown(node, sourceType) {
    const modelWidget = findWidget(node, "model_name");
    if (!modelWidget) return;

    try {
        // Rebuild the server-side model index so newly added files are found
        await DRAGOS_API.rebuildModelIndex();

        const data = await DRAGOS_API.getModelList(sourceType);
        const modelList = data.models || [];

        // Store the list for the custom model picker dialog
        node._dragosModelList = modelList;

        if (modelList.length === 0) {
            modelWidget.value = "";

            if (node._dragosModelDisplayWidget) {
                node._dragosModelDisplayWidget.value = "(no models found)";
            }

            const settingWidget = findWidget(node, "saved_setting");
            if (settingWidget) {
                updateComboWidget(settingWidget, ["(none)"], "(none)");
            }

            node.setDirtyCanvas(true, true);
            app.graph.setDirtyCanvas(true, true);
            return;
        }

        const currentModel = modelWidget.value;

        const selectedModel =
            modelList.includes(currentModel)
                ? currentModel
                : modelList[0];

        modelWidget.value = selectedModel;

        if (node._dragosModelDisplayWidget) {
            node._dragosModelDisplayWidget.value = selectedModel;
        }

        node.setDirtyCanvas(true, true);
        app.graph.setDirtyCanvas(true, true);

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

        // Update saved_setting dropdown in-place
        const settingNames = Object.keys(settings);
        if (settingWidget) {
            if (settingNames.length > 0) {
                const currentSetting = settingWidget.value;
                const defaultSetting = settingNames.includes(currentSetting) ? currentSetting : settingNames[0];
                updateComboWidget(settingWidget, settingNames, defaultSetting);
                // Auto-load the default setting
                await loadSettingValues(node, defaultSetting);
            } else {
                // No saved settings - show placeholder and load base defaults
                updateComboWidget(settingWidget, ["(none)"], "(none)");

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

    if (!sourceType || !modelName || modelName === "(no models found)") {
        alert("Please select a valid model source and model before saving.");
        return;
    }

    // If saved_setting is "(none)" or empty, prompt for a name
    let finalSettingName = settingName;
    if (!settingName || settingName === "(none)") {
        finalSettingName = await promptForSettingName();
        if (!finalSettingName) return; // User cancelled
    }

    try {
        const result = await DRAGOS_API.saveModelSetting({
            source_type: sourceType,
            model_name: modelName,
            setting_name: finalSettingName,
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
            // Refresh the settings dropdown and select the saved setting
            await loadModelMetadata(node);
            const settingWidget = findWidget(node, "saved_setting");
            if (settingWidget) {
                setWidgetValue(settingWidget, finalSettingName);
            }
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

    if (!sourceType || !modelName || modelName === "(no models found)") {
        alert("Please select a valid model source and model before saving a new setting.");
        return;
    }

    // Use custom dialog instead of browser prompt() which can be blocked
    const newSettingName = await promptForSettingName();
    if (!newSettingName) return; // User cancelled

    try {
        const result = await DRAGOS_API.saveNewModelSetting({
            source_type: sourceType,
            model_name: modelName,
            new_setting_name: newSettingName,
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
                setWidgetValue(settingWidget, newSettingName);
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

    if (!sourceType || !modelName || !settingName || settingName === "(none)") {
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
		setTimeout(async () => {
			await loadLoraMetadata(node, isModelOnly);
		}, 100);
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

        // Listen for ComfyUI refresh (R key) and rebuild the model index
        api.addEventListener("status", async () => {
            try {
                await DRAGOS_API.rebuildModelIndex();
            } catch (e) {
                // Silently ignore — not all status events are refreshes
            }
        });
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
