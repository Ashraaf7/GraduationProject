using CourseApp1.data;
using CourseApp1.Models;
using Microsoft.AspNet.Identity;
using Microsoft.AspNet.Identity.EntityFramework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Web;
using System.Web.Mvc;

namespace CourseApp1.Controllers
{
    public class AccountController : Controller
    {
        private readonly UserManager<MyIdentityUser> userManager;
        private courses_dbEntities db = new courses_dbEntities();
        public AccountController()
        {
            var db = new CoursesIdentity();
            var userStore=new UserStore<MyIdentityUser>(db);
            userManager = new UserManager<MyIdentityUser>(userStore);
        }
        // GET: Account
        [AllowAnonymous]
        public ActionResult login(String ReturnUrl)
        {
            
            return View(new LoginViewModel
            {
                ReturnUrl=ReturnUrl
            });
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<ActionResult> login(LoginViewModel logindata)
        {
            if (ModelState.IsValid)
            {
                var exist = await userManager.FindAsync(logindata.Email, logindata.Password);
                if(exist != null)
                {
                    await SignIn(exist);
                    if (!string.IsNullOrEmpty(logindata.ReturnUrl))
                    {
                        return Redirect(logindata.ReturnUrl);
                    }
                    var userRole = userManager.GetRoles(exist.Id);
                    var role = userRole.FirstOrDefault();
                    if(role == "Admin")
                    {
                        return RedirectToAction("Index", "Default", new { area = "Admin" });
                    }
                    return RedirectToAction("Index", "Default");
                }
                logindata.message = "Email or Password incorrect";
            }
            return View(logindata);
        }
        public async Task SignIn(MyIdentityUser myIdentityUser)
        {
            var identity = await userManager.CreateIdentityAsync(myIdentityUser, DefaultAuthenticationTypes.ApplicationCookie);
            var owinContext = Request.GetOwinContext();
            var authmanager = owinContext.Authentication;
            authmanager.SignIn(identity);
        }
        [AllowAnonymous]
        public ActionResult Register()
        {
            return View();
        }
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<ActionResult> Register(RegisterViewModel userinfo)
        {
            if (ModelState.IsValid)
            {
                var identityuser = new MyIdentityUser()
                {
                    Email = userinfo.Email,
                    UserName = userinfo.Email
                };
                var result = await userManager.CreateAsync(identityuser,userinfo.Password);
                if (result.Succeeded)
                {
                    var userId = identityuser.Id;
                    result=userManager.AddToRole(userId, "Trainee");
                    if (result.Succeeded)
                    {
                        db.Trainees.Add(new Trainee { Email = userinfo.Email, Name = userinfo.Name });
                        int saving=db.SaveChanges();
                        if (saving ==0)
                        {
                            userinfo.message = "An error occured while creating your account";
                            return View(userinfo); 
                        }
                        return RedirectToAction("Index", "Default");
                    }
                    
                }
                else
                {
                    var message=result.Errors.FirstOrDefault();
                    userinfo.message=message;
                    return View(userinfo);
                }
            }
            return View(userinfo);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public ActionResult Logout()
        {
            var owinContext = Request.GetOwinContext();
            var authmanager = owinContext.Authentication;
            authmanager.SignOut("ApplicationCookie");
            Session.Abandon();
            return RedirectToAction("Index","Default");
        }
    }
}