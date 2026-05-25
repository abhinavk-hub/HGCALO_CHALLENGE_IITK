#z_dim = 28
x_dim = 4       
y_dim = 3
hidden_dim = 128

#encoder = GraphEncoder(x_dim, y_dim, z_dim, hidden_dim).to(device)
#decoder = GraphDecoder(z_dim, y_dim, x_dim, hidden_dim).to(device)
#opt_encdec = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

E_flow = TotalEnergyFlow(y_dim).to(device)
layer_flow = LayerGraphFlow(feat_dim=6, cond_dim=y_dim+1, hidden_dim= 64, num_layers=4).to(device)
node_flow = NodeGraphFlow(x_dim, y_dim, hidden_dim, layer_hidden=64, time_dim=32, num_layers=4).to(device)
opt_ef = torch.optim.Adam(E_flow.parameters(), lr=1e-4)
opt_lf = torch.optim.Adam(layer_flow.parameters(), lr=1e-4)
opt_nf = torch.optim.Adam(node_flow.parameters(), lr=1e-4)
#scheduler = MultiStepLR(opt_nf, milestones=[20, 30], gamma=0.1)
scheduler = StepLR(opt_nf, step_size=20, gamma=0.1)
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")
'''
for epoch in range(10):
    encoder.train()
    decoder.train()
    total = 0
    for batch in train_loader:
        batch = batch.to(device)
        y = batch.y.view(batch.num_graphs, y_dim)

        z = encoder(batch.x, batch.edge_index, y, batch.batch)
        x_hat = decoder(z, batch.edge_index, y, batch.batch)

        loss = F.mse_loss(x_hat, batch.x)

        opt_encdec.zero_grad()
        loss.backward()
        opt_encdec.step()
        total += loss.item()

    print(f"[AE] Epoch {epoch} loss {total/len(train_loader):.4f}")

encoder.eval()
decoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)
for p in decoder.parameters():
    p.requires_grad_(False)

def test(E_flow, layer_flow, node_flow, loader, opt_ef, opt_lf, opt_nf, device, alpha):
    E_flow.eval()
    layer_flow.eval()
    node_flow.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            data = batch.to(device)
            x = data.x
            y = data.y.view(data.num_graphs, y_dim)
            edge_index = data.edge_index
            batch = data.batch
            e_tot = data.E_sum.view(data.num_graphs, 1)
            e_r = data.E_ratio.view(data.num_graphs, 1)
            layer_x = data.layer_x
            layer_edge_index = data.layer_edge_index
            layer_idx = data.layer_idx
            num_graphs = data.num_graphs
            nodes_per_graph_l = MAX_LAYERS

            layer_batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph_l)

        
            L_r = E_flow(e_r, y).mean()
            
            E_r_gen = E_flow.sample(y)
            E_r_gen_un = E_r_gen*E_ratio_std + E_ratio_mean
            logE_inc = y[:,0].unsqueeze(-1) * y0_s + y0_m
            E_inc = torch.exp(logE_inc)
            E_tot_gen_un = torch.log(E_r_gen_un*E_inc)
            #E_tot_gen_un = E_r_gen_un*(y[:,0].unsqueeze(-1)*y0_s + y0_m)
            e_tot_gen = (E_tot_gen_un - E_tot_mean)/E_tot_std
            
            L_layer = layer_flow(layer_x, layer_edge_index, layer_batch, y, e_tot).mean()
            
            layer_gen = layer_flow.sample(layer_x.size(0), layer_edge_index, layer_batch, y, e_tot)
            
            E_cond = (1 - alpha) * e_tot + alpha * e_tot_gen.detach()
            l_cond = (1 - alpha) * layer_x + alpha * layer_gen.detach()
        
            #with torch.no_grad():
            #    z0 = encoder(x, edge_index, y, batch)
            z0 = x
            #z0 = (z0 - z_mean) / z_std
            z1 = torch.randn_like(z0)
            t = torch.rand(y.size(0), 1, device=device)
            t_safe = t.clamp(max=1.0 - 0.001)
            t_node = t_safe[batch]
            z_t = (1.0 - t_node) * z1 + t_node * z0
            v_target = (z0 - z_t)/ (1.0 - t_node + 1e-6)
            v_pred = node_flow(z_t, edge_index, batch, y, E_cond, l_cond, layer_edge_index, layer_batch, t_safe)
        
            L_node = ((v_pred - v_target)**2).mean()
            total_loss+= (L_node.item()+L_layer.item()+L_r.item())
    return total_loss/len(loader)            
'''
max_epoch=20
t_loss = []
v_loss = []
tes_loss = []

for epochs in range(1, max_epoch+1):
    if epochs < 6:
        alpha = 0
    elif epochs > 5 and epochs < 16:
        alpha = (epochs - 5)/10
    elif epochs > 15:
        alpha = 1
    E_flow.train()
    layer_flow.train()
    node_flow.train()
    freeze_conditioners = (epochs > 15)

    if freeze_conditioners:
        for p in E_flow.parameters():
            p.requires_grad_(False)
        for p in layer_flow.parameters():
            p.requires_grad_(False)
    else:
        for p in E_flow.parameters():
            p.requires_grad_(True)
        for p in layer_flow.parameters():
            p.requires_grad_(True)
    train_loss = 0
    for batch in train_loader:
        data = batch.to(device)
        x = data.x
        y = data.y.view(data.num_graphs, y_dim)
        edge_index = data.edge_index
        batch = data.batch
        e_tot = data.E_sum.view(data.num_graphs, 1)
        e_r = data.E_ratio.view(data.num_graphs, 1)
        layer_x = data.layer_x
        layer_edge_index = data.layer_edge_index
        layer_idx = data.layer_idx
        num_graphs = data.num_graphs
        nodes_per_graph_l = MAX_LAYERS

        layer_batch = torch.arange(num_graphs, device=device).repeat_interleave(nodes_per_graph_l)

                
        L_r = E_flow(e_r, y).mean()
        if not freeze_conditioners:
            opt_ef.zero_grad()
            L_r.backward()
            opt_ef.step()
        
        E_flow.eval()
        with torch.no_grad():
            E_r_gen = E_flow.sample(y)
            E_r_gen_un = E_r_gen*E_ratio_std + E_ratio_mean
            logE_inc = y[:,0].unsqueeze(-1) * y0_s + y0_m
            E_inc = torch.exp(logE_inc)
            safe_val = (E_r_gen_un * E_inc).clamp_min(1e-8)
            E_tot_gen_un = torch.log(safe_val)
            #E_tot_gen_un = E_r_gen_un*(y[:,0].unsqueeze(-1)*y0_s + y0_m)
            e_tot_gen = (E_tot_gen_un - E_tot_mean)/E_tot_std
        E_flow.train()
        
        L_layer = layer_flow(layer_x, layer_edge_index, layer_batch, y, e_tot).mean()
        if not freeze_conditioners:
            opt_lf.zero_grad()
            L_layer.backward()
            opt_lf.step()
        
        layer_flow.eval()
        with torch.no_grad():
            layer_gen = layer_flow.sample(layer_x.size(0), layer_edge_index, layer_batch, y, e_tot)
        layer_flow.train()
        
        E_cond = (1 - alpha) * e_tot + alpha * e_tot_gen.detach()
        l_cond = (1 - alpha) * layer_x + alpha * layer_gen.detach()
        
        #with torch.no_grad():
        #    z0 = encoder(x, edge_index, y, batch)
        z0 = x
        #z0 = (z0 - z_mean) / z_std
        z1 = torch.randn_like(z0)
        t = torch.rand(y.size(0), 1, device=device)
        t_safe = t.clamp(max=1.0 - 0.001)
        t_node = t_safe[batch]
        z_t = (1.0 - t_node) * z1 + t_node * z0
        v_target = (z0 - z_t)/ (1.0 - t_node + 1e-3)
        v_pred = node_flow(z_t, edge_index, batch, y, E_cond, l_cond, layer_edge_index, layer_batch, t_safe)
        
        L_node = ((v_pred - v_target)**2).mean()
        opt_nf.zero_grad()
        L_node.backward()
        opt_nf.step()
        train_loss+= (L_node.item()+L_layer.item()+L_r.item())
    train_loss/=len(train_loader)
    t_loss.append(train_loss)
    #val_loss = test(E_flow, layer_flow, node_flow, val_loader, opt_ef, opt_lf, opt_nf, device, alpha)
    #v_loss.append(val_loss)
    #test_loss = test(E_flow, layer_flow, node_flow, test_loader, opt_ef, opt_lf, opt_nf, device, alpha)
    #tes_loss.append(test_loss)
    #print(f"Epoch {epochs:03d} TrainLoss {train_loss:.4f} val_loss {val_loss:.4f} Test_loss {test_loss:.4f}")
    print(f"Epoch {epochs:03d} TrainLoss {train_loss:.4f}")
    #scheduler.step()

plt.plot(t_loss, label="Train")
#plt.plot(v_loss, label="Validation")
#plt.plot(tes_loss, label="Test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.show()
