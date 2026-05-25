opt_ef = torch.optim.Adam(energy_flow.parameters(), lr= 1e-4)
opt_lf = torch.optim.Adam(long_flow.parameters(), lr= 1e-4)
opt_rfm = torch.optim.Adam(trans_rfm.parameters(), lr= 1e-4)

epoch_energy = 5
epoch_long = 10
epoch_trans = 20

def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

def unfreeze(model):
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

def unpack(batch, device):
    batch = batch.to(device)
    
    E_inc   = batch.E_inc.view(-1, 1)
    E_total = batch.E_total.view(-1, 1)

    long_x          = batch.long_x              
    long_edge_index = batch.long_edge_index     
    long_batch      = batch.long_x_batch               

    raw_nhits = (long_x[:, 1] * stats['long_n_std'] + stats['long_n_mean'])
    raw_nhits = (torch.exp(raw_nhits) - 1).round().long().clamp(min=0)

    trans_x          = batch.trans_x            
    trans_edge_index = batch.trans_edge_index   
    trans_batch      = batch.trans_batch
    trans_x_batch    = batch.trans_x_batch
    trans_layer_ids  = batch.trans_layer_ids    
    trans_E_l        = batch.trans_E_l          
    trans_nhits      = batch.trans_nhits        

    return (E_inc, E_total, long_x, long_edge_index, long_batch, raw_nhits, trans_x, trans_edge_index, trans_batch, trans_layer_ids, trans_E_l, trans_nhits)

print("PHASE 1 — EnergyFlow")

ef_train_losses = []
ef_val_losses   = []

unfreeze(energy_flow)

for epoch in range(1, epoch_energy + 1):

    energy_flow.train()
    train_loss = 0.0

    for batch in train_loader:
        (E_inc, E_total,
         long_x, long_edge_index, long_batch, raw_nhits,
         trans_x, trans_edge_index, trans_batch,
         trans_layer_ids, trans_E_l, trans_nhits) = unpack(batch, device)

        L = energy_flow(E_total, E_inc).mean()

        opt_ef.zero_grad()
        L.backward()
        opt_ef.step()
        train_loss += L.item()

    train_loss /= len(train_loader)

    energy_flow.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            (E_inc, E_total,long_x, long_edge_index, long_batch, raw_nhits,trans_x, trans_edge_index, trans_batch, trans_layer_ids, trans_E_l, trans_nhits) = unpack(batch, device)
            val_loss += energy_flow(E_total, E_inc).mean().item()

    val_loss /= len(val_loader)

    ef_train_losses.append(train_loss)
    ef_val_losses.append(val_loss)
    print(f"  [EnergyFlow] Epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f}")

freeze(energy_flow)
print("EnergyFlow frozen.")

print("PHASE 2 — LongGraphFlow")

lf_train_losses  = []
lf_val_losses    = []
poi_train_losses = []
poi_val_losses   = []

unfreeze(long_flow)

for epoch in range(1, epoch_long + 1):

    long_flow.train()
    train_loss_e = 0.0
    train_loss_p = 0.0

    for batch in train_loader:
        (E_inc, E_total,
         long_x, long_edge_index, long_batch, raw_nhits,
         trans_x, trans_edge_index, trans_batch,
         trans_layer_ids, trans_E_l, trans_nhits) = unpack(batch, device)

        with torch.no_grad():
            E_total_sampled = energy_flow.sample(E_inc)     

        nll_energy, nll_poisson = long_flow(
            long_x, long_edge_index, long_batch,
            E_total_sampled.squeeze(-1),                     
            E_inc.squeeze(-1),                               
            raw_nhits
        )
        
        nll_e = nll_energy.mean() / (nll_energy.mean().abs().detach().clamp(min=1e-6))
        nll_p = nll_poisson.mean() / (nll_poisson.mean().abs().detach().clamp(min=1e-6))
        L = nll_e + nll_p

        opt_lf.zero_grad()
        L.backward()
        torch.nn.utils.clip_grad_norm_(long_flow.parameters(), 1.0)
        opt_lf.step()
# using normalized loss only for training and displaying that.
        train_loss_e += nll_energy.mean().item()
        train_loss_p += nll_poisson.mean().item()

    train_loss_e /= len(train_loader)
    train_loss_p /= len(train_loader)

    long_flow.eval()
    val_loss_e = 0.0
    val_loss_p = 0.0

    with torch.no_grad():
        for batch in val_loader:
            (E_inc, E_total,long_x, long_edge_index, long_batch, raw_nhits,trans_x, trans_edge_index, trans_batch,trans_layer_ids, trans_E_l, trans_nhits) = unpack(batch, device)

            E_total_sampled = energy_flow.sample(E_inc)

            nll_e, nll_p = long_flow(
                long_x, long_edge_index, long_batch,
                E_total_sampled.squeeze(-1),
                E_inc.squeeze(-1),
                raw_nhits
            )
            val_loss_e += nll_e.mean().item()
            val_loss_p += nll_p.mean().item()

    val_loss_e /= len(val_loader)
    val_loss_p /= len(val_loader)

    lf_train_losses.append(train_loss_e)
    lf_val_losses.append(val_loss_e)
    poi_train_losses.append(train_loss_p)
    poi_val_losses.append(val_loss_p)

    print(f"  [LongFlow] Epoch {epoch:03d}  "
          f"energy train {train_loss_e:.4f}  val {val_loss_e:.4f}  |  "
          f"poisson train {train_loss_p:.4f}  val {val_loss_p:.4f}")

freeze(long_flow)
print("LongGraphFlow frozen.")

print("PHASE 3 — TransRFM")

rfm_train_losses = []
rfm_val_losses   = []

unfreeze(trans_rfm)

for epoch in range(1, epoch_trans + 1):

    trans_rfm.train()
    train_loss = 0.0

    for batch in train_loader:
        (E_inc, E_total,
         long_x, long_edge_index, long_batch, raw_nhits,
         trans_x, trans_edge_index, trans_batch,
         trans_layer_ids, trans_E_l, trans_nhits) = unpack(batch, device)

        B = E_inc.size(0)

        with torch.no_grad():
            E_total_sampled = energy_flow.sample(E_inc)  

            log_E_l_sampled, nhits_sampled, active_sampled = long_flow.sample(
                long_edge_index,
                long_batch,
                E_total_sampled.squeeze(-1),                        
                E_inc.squeeze(-1)                                    
            )

        shower_of_layergraph = torch.repeat_interleave(torch.arange(B, device=device), batch.n_active)
        long_node_idx = shower_of_layergraph * MAX_LAYERS + trans_layer_ids
        E_l_cond    = log_E_l_sampled[long_node_idx]               
        nhits_cond  = nhits_sampled[long_node_idx].float()         
        nhits_cond_norm = (torch.log(nhits_cond + 1) - stats['tn_mean']) / stats['tn_std']

        z0 = trans_x                                               
        z1 = torch.randn_like(z0)                                  
        t_scalar = torch.rand(1, device=device).item()
        t = torch.full((E_l_cond.size(0), 1), t_scalar, device=device)        
        t_node = t[trans_batch]                                   
        z_t      = (1.0 - t_node) * z1 + t_node * z0
        if t_scalar >= 0.4:
            ei_used = trans_edge_index
        else:
            ei_used = torch.zeros((2, 0), dtype=torch.long, device=device)
        v_target = z0 - z1                                         
        v_pred = trans_rfm(
            z_t, ei_used, trans_batch,
            E_l_cond, nhits_cond_norm,
            trans_layer_ids, t
        )
        L = ((v_pred - v_target) ** 2).mean()

        opt_rfm.zero_grad()
        L.backward()
        opt_rfm.step()
        train_loss += L.item()

    train_loss /= len(train_loader)

    trans_rfm.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            (E_inc, E_total,
             long_x, long_edge_index, long_batch, raw_nhits,
             trans_x, trans_edge_index, trans_batch,
             trans_layer_ids, trans_E_l, trans_nhits) = unpack(batch, device)

            B = E_inc.size(0)

            E_total_sampled = energy_flow.sample(E_inc)
            log_E_l_sampled, nhits_sampled, active_sampled = long_flow.sample(
                long_edge_index, long_batch,
                E_total_sampled.squeeze(-1),
                E_inc.squeeze(-1)
            )

            shower_of_layergraph = torch.repeat_interleave(torch.arange(B, device=device), batch.n_active)
            long_node_idx = shower_of_layergraph * MAX_LAYERS + trans_layer_ids
            E_l_cond      = log_E_l_sampled[long_node_idx]
            nhits_cond    = nhits_sampled[long_node_idx].float()
            nhits_cond_norm = (torch.log(nhits_cond + 1) - stats['tn_mean']) / stats['tn_std']

            z0 = trans_x
            z1 = torch.randn_like(z0)
            t_scalar = torch.rand(1, device=device).item()
            t = torch.full((E_l_cond.size(0), 1), t_scalar, device=device)
            t_node   = t[trans_batch]
            z_t      = (1.0 - t_node) * z1 + t_node * z0
            if t_scalar >= 0.4:
                ei_used = trans_edge_index
            else:
                ei_used = torch.zeros((2, 0), dtype=torch.long, device=device)
            v_target = z0 - z1
            v_pred = trans_rfm(
                z_t, ei_used, trans_batch,
                E_l_cond, nhits_cond_norm,
                trans_layer_ids, t
            )

            val_loss += ((v_pred - v_target) ** 2).mean().item()

    val_loss /= len(val_loader)

    rfm_train_losses.append(train_loss)
    rfm_val_losses.append(val_loss)
    print(f"  [TransRFM] Epoch {epoch:03d}  train {train_loss:.4f}  val {val_loss:.4f}")

freeze(trans_rfm)
print("TransRFM frozen. Training complete.")

torch.save(energy_flow.state_dict(), "energy_flow.pt")
torch.save(long_flow.state_dict(),   "long_flow.pt")
torch.save(trans_rfm.state_dict(),   "trans_rfm.pt")
print("Models saved.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(ef_train_losses,  label="Train")
axes[0].plot(ef_val_losses,    label="Val")
axes[0].set_title("EnergyFlow")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("NLL")
axes[0].legend()

axes[1].plot(lf_train_losses,  label="Energy Train")
axes[1].plot(lf_val_losses,    label="Energy Val")
axes[1].plot(poi_train_losses, label="Poisson Train", linestyle='--')
axes[1].plot(poi_val_losses,   label="Poisson Val",   linestyle='--')
axes[1].set_title("LongGraphFlow")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("NLL")
axes[1].legend()

axes[2].plot(rfm_train_losses, label="Train")
axes[2].plot(rfm_val_losses,   label="Val")
axes[2].set_title("TransRFM")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("MSE")
axes[2].legend()

plt.tight_layout()
#plt.savefig("loss_curves.png", dpi=150)
plt.show()
print("Loss curves saved.")
