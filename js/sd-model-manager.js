import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js"

console.log("[sd-model-manager]", "Loading js extension");
app.registerExtension({
	name: "Comfy.SDModelManagerMenu",
	init() {
	},
	async setup() {
		const menu = document.querySelector(".comfy-menu");
		const managerButton = document.createElement("button");
		managerButton.textContent = "Model Manager";
		managerButton.onclick = () => {
            open_manager();
        }
		menu.append(managerButton);
	},
});

async function open_manager() {
	api.fetchApi('/models/api/v1/open_manager')
}